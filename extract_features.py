"""
Extract and cache TAMT frozen features for ablation experiments.

Why this exists: each test.py eval spends ~95% of wall time on video
decoding + VideoMAE forward. The actual test-time method (CTPR / TCR / etc.)
is <5%. Caching features once means subsequent ablation configs run in
minutes instead of hours.

Per (dataset, n_shot) cache:
  - Iterate test_n_episode × test_task_nums episodes using the SAME loader
    config as test.py (seed=42, n_query=12).
  - For each episode, run parse_feature → (support_feats, query_feats).
  - Concatenate everything into two big tensors + save to .pt.

Cache format (for one dataset+shot):
  {
    'dataset': str, 'n_shot': int, 'n_way': int, 'n_query': int,
    'n_episodes': int, 'n_tasks': int, 'seed': int,
    'support_feats': Tensor [N_eps, n_way*n_shot, D],
    'query_feats':   Tensor [N_eps, n_way*n_query, D],
    'ckpt_path': str, 'ckpt_md5': str,
  }

Usage:
  python extract_features.py \
      --dataset SSv2Full --data_path filelist/SSv2Full \
      --model_path checkpoints_official/ssv2/best_model.tar \
      --n_shot 1 --test_n_episode 2000 --test_task_nums 5 \
      --out cached_feats/SSv2Full_1shot.pt

Fairness/correctness:
  - Uses exactly the same SetDataManager + seed as test.py — so the cached
    episodes are byte-for-byte the SAME 2000 ep × 5 tasks that test.py would
    have sampled. This is critical: methods evaluated on cached features
    yield identical numbers to running them via test.py.
  - Stores ckpt MD5 — fast_eval checks it matches before loading, preventing
    silent cache-miss when a checkpoint is swapped.
"""
import os, time, json, hashlib, argparse
import numpy as np
import torch
import tqdm

from data.datamgr import SetDataManager
from methods.meta_deepbdc import MetaDeepBDC
from methods.protonet import ProtoNet
from utils import model_dict, load_model, set_gpu


def md5_of_file(path, chunk=4 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(chunk), b''):
            h.update(c)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_path', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--model', default='VideoMAES')
    p.add_argument('--method', default='meta_deepbdc')
    p.add_argument('--image_size', default=112, type=int)
    p.add_argument('--reduce_dim', default=256, type=int)
    p.add_argument('--head_variant', default='tamt')
    p.add_argument('--n_way', default=5, type=int)
    p.add_argument('--n_shot', default=1, type=int)
    p.add_argument('--n_query', default=12, type=int)
    p.add_argument('--test_n_episode', default=2000, type=int)
    p.add_argument('--test_task_nums', default=5, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--gpu', default='0')
    p.add_argument('--out', required=True)
    # alias to mimic test.py's MetaDeepBDC constructor signature
    p.add_argument('--penalty_C', default=0.1, type=float)
    p.add_argument('--dropout_rate', default=0.5, type=float)
    args = p.parse_args()

    # ---- seed (mirror test.py exactly) ----
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_gpu(args)
    print(f'[SEED] {args.seed}')

    # ---- novel_file (mirror test.py dispatch) ----
    novel_file = 'novel.json'

    # ---- Data loader ----
    dm = SetDataManager(args.data_path, args.image_size,
                        n_query=args.n_query, n_episode=args.test_n_episode,
                        json_read=True, n_way=args.n_way,
                        n_support=args.n_shot)
    loader = dm.get_data_loader(novel_file, aug=False)

    # ---- Model (method-aware: protonet/good_embed use avgpool, no BDC) ----
    if args.method in ('protonet', 'good_embed'):
        model = ProtoNet(args, model_dict[args.model],
                         n_way=args.n_way, n_support=args.n_shot).cuda()
    else:  # meta_deepbdc (TAMT)
        model = MetaDeepBDC(args, model_dict[args.model],
                            n_way=args.n_way, n_support=args.n_shot).cuda()
    model.eval()
    model = load_model(model, args.model_path)

    n_total = args.test_n_episode * args.test_task_nums
    print(f'[EXTRACT] {args.dataset} {args.n_shot}-shot, '
          f'{args.test_n_episode} ep × {args.test_task_nums} tasks = {n_total}')

    sup_all = []
    qry_all = []
    t0 = time.time()

    # The original test.py loops the loader `test_task_nums` times. Replicate.
    for task_idx in range(args.test_task_nums):
        tqdm_gen = tqdm.tqdm(loader, total=args.test_n_episode,
                              desc=f'task {task_idx+1}/{args.test_task_nums}')
        for ep_idx, (x, _) in enumerate(tqdm_gen):
            x = x.cuda()
            model.n_query = args.n_query
            with torch.no_grad():
                out = model.parse_feature(x, is_feature=False)
                # protonet returns (z_s, z_q); meta_deepbdc returns (z_s, z_q, _, _)
                z_s, z_q = out[0], out[1]
            sup = z_s.contiguous().view(
                args.n_way * args.n_shot, -1).detach().cpu()
            qry = z_q.contiguous().view(
                args.n_way * args.n_query, -1).detach().cpu()
            sup_all.append(sup)
            qry_all.append(qry)

    sup_t = torch.stack(sup_all, dim=0)   # [N_eps, n_way*n_shot, D]
    qry_t = torch.stack(qry_all, dim=0)   # [N_eps, n_way*n_query, D]
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    # Safety guard: refuse to silently overwrite a cache built from a
    # DIFFERENT method/ckpt (this is exactly the bug class we must avoid).
    if os.path.exists(args.out):
        old = torch.load(args.out, map_location='cpu', weights_only=False)
        if old.get('method', 'meta_deepbdc') != args.method or \
           old.get('ckpt_path') != args.model_path:
            raise RuntimeError(
                f'REFUSING to overwrite {args.out}: existing cache was built '
                f'from method={old.get("method","meta_deepbdc")} '
                f'ckpt={old.get("ckpt_path")}, but this run is '
                f'method={args.method} ckpt={args.model_path}. '
                f'Use a different --out name.')
    payload = {
        'dataset': args.dataset, 'n_shot': args.n_shot, 'method': args.method,
        'n_way': args.n_way, 'n_query': args.n_query,
        'n_episodes': args.test_n_episode, 'n_tasks': args.test_task_nums,
        'seed': args.seed,
        'support_feats': sup_t, 'query_feats': qry_t,
        'ckpt_path': args.model_path, 'ckpt_md5': md5_of_file(args.model_path),
    }
    torch.save(payload, args.out)
    print(f'[EXTRACT] saved {sup_t.shape} support, {qry_t.shape} query → {args.out}')
    print(f'[EXTRACT] elapsed: {elapsed/60:.1f} min '
          f'({elapsed/n_total:.2f} s/ep)')


if __name__ == '__main__':
    main()
