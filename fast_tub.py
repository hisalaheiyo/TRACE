"""
Cache-based TUB oracle (fast). Recomputes the three transductive upper bounds
on PRE-EXTRACTED cached features (extract_features.py), so it uses the SAME
features as fast_eval and is correct w.r.t. the self-trained 1-shot ckpt.

Reuses the exact oracle logic from tub_clf_oracle.py:
  - acc_base_halfB   : prototype baseline on held-out half_B
  - acc_proto_oracle : TUB-Proto (support ∪ half_A true labels, TAMT metric)
  - acc_tub_clf      : TUB-Clf  (LR on support ∪ half_A, eval half_B)
  - acc_tcr_oracle_init : TUB-TCR-Init (TCR init from oracle proto, eval half_B)

Usage:
  python fast_tub.py --cache cached_feats/hmdb51_1shot.pt \\
      --n_splits 20 --max_episodes 3000
"""
import argparse, json, time, os
import numpy as np
import torch
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', category=ConvergenceWarning)

from methods.tcr import tcr_refine


def proto_score(s_feats, s_labels, q_feats, n_way, metric):
    prots = torch.zeros(n_way, s_feats.shape[-1],
                        device=s_feats.device, dtype=s_feats.dtype)
    for k in range(n_way):
        m = (s_labels == k)
        if m.sum() > 0:
            prots[k] = s_feats[m].mean(0)
    q = q_feats.unsqueeze(1); p = prots.unsqueeze(0)
    if metric == 'l2':
        return -((q - p) ** 2).sum(-1)
    return (q * p).sum(-1)   # dot


def lr_eval(train_x, train_y, eval_x, eval_y, seed):
    eps = 1e-12
    tx = train_x / (np.linalg.norm(train_x, axis=1, keepdims=True) + eps)
    ex = eval_x  / (np.linalg.norm(eval_x,  axis=1, keepdims=True) + eps)
    clf = LogisticRegression(C=10.0, max_iter=1000, solver='lbfgs',
                             random_state=seed, n_jobs=1)
    clf.fit(tx, train_y)
    return float((clf.predict(ex) == eval_y).mean())


def episode_oracles(s_feats, q_feats, n_way, n_shot, n_query, metric,
                    n_splits, ep_seed, device):
    s_feats = s_feats.to(device); q_feats = q_feats.to(device)
    s_labels = torch.arange(n_way, device=device).repeat_interleave(n_shot)
    q_labels = torch.arange(n_way, device=device).repeat_interleave(n_query)

    sc_base = proto_score(s_feats, s_labels, q_feats, n_way, metric)
    acc_base_all = (sc_base.argmax(1) == q_labels).float().mean().item()

    rng = np.random.RandomState(ep_seed)
    s_np = s_feats.cpu().numpy(); q_np = q_feats.cpu().numpy()
    sl_np = s_labels.cpu().numpy(); ql_np = q_labels.cpu().numpy()
    N_q = q_feats.shape[0]

    base_s, proto_s, tub_s, tinit_s = [], [], [], []
    for si in range(n_splits):
        perm = rng.permutation(N_q); half = N_q // 2
        iA, iB = perm[:half], perm[half:]
        qa_f, qa_y = q_np[iA], ql_np[iA]
        qb_f, qb_y = q_np[iB], ql_np[iB]

        base_s.append(float((sc_base[iB].argmax(1).cpu().numpy() == qb_y).mean()))

        tr_f = torch.from_numpy(np.concatenate([s_np, qa_f], 0)).to(device)
        tr_y = torch.from_numpy(np.concatenate([sl_np, qa_y], 0)).to(device)
        qb_t = torch.from_numpy(qb_f).to(device)
        sc_o = proto_score(tr_f, tr_y, qb_t, n_way, metric)
        proto_s.append(float((sc_o.argmax(1).cpu().numpy() == qb_y).mean()))

        tub_s.append(lr_eval(np.concatenate([s_np, qa_f], 0),
                             np.concatenate([sl_np, qa_y], 0),
                             qb_f, qb_y, seed=ep_seed + si))

        # oracle proto for TCR init
        P_or = torch.zeros(n_way, s_feats.shape[-1], device=device,
                           dtype=s_feats.dtype)
        for k in range(n_way):
            m = (tr_y == k)
            if m.sum() > 0:
                P_or[k] = tr_f[m].mean(0)
        with torch.enable_grad():
            tl = tcr_refine(s_feats, s_labels, qb_t, n_way=n_way,
                            n_iter=100, lr=1e-3, alpha=0.1, beta=1.0,
                            temp=15.0, metric='cos', init_W=P_or)
        tinit_s.append(float((tl.argmax(1).cpu().numpy() == qb_y).mean()))

    return {
        'acc_base_all': acc_base_all,
        'acc_base_halfB': float(np.mean(base_s)),
        'acc_proto_oracle': float(np.mean(proto_s)),
        'acc_tub_clf': float(np.mean(tub_s)),
        'acc_tcr_oracle_init': float(np.mean(tinit_s)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--n_splits', default=20, type=int)
    p.add_argument('--max_episodes', default=3000, type=int,
                   help='subsample episodes for speed (0 = all)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out_json', default='')
    args = p.parse_args()

    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    sup = cache['support_feats']; qry = cache['query_feats']
    n_way = cache['n_way']; n_shot = cache['n_shot']; n_query = cache['n_query']
    metric = 'dot' if n_shot == 1 else 'l2'
    N = sup.shape[0]
    if args.max_episodes and args.max_episodes < N:
        N = args.max_episodes

    print(f'[TUB] {cache["dataset"]} {n_shot}-shot, {N} ep × {args.n_splits} '
          f'splits, metric={metric}', flush=True)
    print(f'[TUB] ckpt: {cache.get("ckpt_path","?")}', flush=True)

    keys = ['acc_base_all', 'acc_base_halfB', 'acc_proto_oracle',
            'acc_tub_clf', 'acc_tcr_oracle_init']
    acc = {k: [] for k in keys}
    t0 = time.time()
    for ep in range(N):
        info = episode_oracles(sup[ep], qry[ep], n_way, n_shot, n_query,
                               metric, args.n_splits, ep_seed=42 + ep,
                               device=args.device)
        for k in keys:
            acc[k].append(info[k])
        if (ep + 1) % 500 == 0:
            print(f'  {ep+1}/{N}  base={100*np.mean(acc["acc_base_halfB"]):.2f} '
                  f'protoO={100*np.mean(acc["acc_proto_oracle"]):.2f} '
                  f'TUBclf={100*np.mean(acc["acc_tub_clf"]):.2f} '
                  f'TIMinit={100*np.mean(acc["acc_tcr_oracle_init"]):.2f}', flush=True)

    summary = {k: {'mean': 100*float(np.mean(acc[k])),
                   'sem': 100*float(np.std(acc[k])/np.sqrt(len(acc[k])))}
               for k in keys}
    b = summary['acc_base_halfB']['mean']
    print(f'\n===== TUB Summary: {cache["dataset"]} {n_shot}-shot =====', flush=True)
    print(f'base(halfB)   {b:.2f}', flush=True)
    print(f'proto-oracle  {summary["acc_proto_oracle"]["mean"]:.2f} '
          f'(dP={summary["acc_proto_oracle"]["mean"]-b:+.2f})', flush=True)
    print(f'TUB-Clf       {summary["acc_tub_clf"]["mean"]:.2f} '
          f'(dC={summary["acc_tub_clf"]["mean"]-b:+.2f})', flush=True)
    print(f'TUB-TCR-Init  {summary["acc_tcr_oracle_init"]["mean"]:.2f} '
          f'(dA={summary["acc_tcr_oracle_init"]["mean"]-b:+.2f})', flush=True)
    print(f'elapsed {(time.time()-t0)/60:.1f} min', flush=True)

    out = args.out_json or f'logs/tub_fast_{cache["dataset"]}_{n_shot}shot.json'
    with open(out, 'w') as f:
        json.dump({'dataset': cache['dataset'], 'n_shot': n_shot,
                   'ckpt_path': cache.get('ckpt_path'),
                   'n_episodes': N, 'n_splits': args.n_splits,
                   'summary': summary}, f, indent=2)
    print(f'saved → {out}', flush=True)


if __name__ == '__main__':
    main()
