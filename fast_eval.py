"""
Fast evaluation using cached TAMT features (produced by extract_features.py).

Loads (support_feats, query_feats) tensors from a .pt cache and runs any
of the existing test-time methods directly — skipping video decode + ViT
forward (~95% of normal eval time).

Supported methods:
  - plain         : prototype-only (TAMT metric)
  - ctpr          : CTPR iter3 (or any iter/threshold)
  - tcr           : TCR only
  - tcr_ctpr_ensemble      : 50/50 ensemble (or any tcr_ctpr_w)
  - trace : TRACE (blendable via ctpr_init_w)

Usage:
  python fast_eval.py --cache cached_feats/SSv2Full_1shot.pt \\
      --method trace --ctpr_init_w 0.7 \\
      --ctpr_iter 3 --ctpr_threshold 0.5 \\
      --tcr_iter 100 --tcr_alpha 0.1 --tcr_beta 1.0

Verification: passing --method plain with no other args reproduces test.py's
baseline accuracy to within float precision (cache is deterministic).
"""
import argparse, time, json
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from methods.ctpr import compute_prototypes, score_with_prototypes, ctpr_refine
from methods.tcr import tcr_refine


def run_episode(s_feats, q_feats, n_way, n_shot, method, args, device='cuda'):
    """Run one method on one episode's features. Returns logits [N_q, n_way]."""
    s_feats = s_feats.to(device)
    q_feats = q_feats.to(device)
    s_labels = torch.arange(n_way, device=device).repeat_interleave(n_shot)
    metric = 'dot' if n_shot == 1 else 'l2'

    if method == 'plain':
        proto = compute_prototypes(s_feats, s_labels, n_way)
        return score_with_prototypes(q_feats, proto, metric=metric)

    if method == 'ctpr':
        scores, _ = ctpr_refine(
            s_feats, s_labels, q_feats, n_way=n_way, metric=metric,
            conf_threshold=args.ctpr_threshold, n_iter=args.ctpr_iter)
        return scores

    if method == 'tcr':
        return tcr_refine(
            s_feats, s_labels, q_feats, n_way=n_way,
            n_iter=args.tcr_iter, lr=args.tcr_lr,
            alpha=args.tcr_alpha, beta=args.tcr_beta,
            temp=args.tcr_temp, metric=args.tcr_metric)

    if method in ('tcr_ctpr_ensemble', 'trace', 'trace_ens'):
        # CTPR refined proto (used as ensemble component for tcr_ctpr_ensemble,
        # and as TCR init for trace / trace_ens)
        proto = compute_prototypes(s_feats, s_labels, n_way)
        for _ in range(args.ctpr_iter):
            sc = score_with_prototypes(q_feats, proto, metric=metric)
            conf, pl = F.softmax(sc, dim=-1).max(dim=-1)
            hc = conf > args.ctpr_threshold
            if hc.sum() > 0:
                proto = compute_prototypes(
                    torch.cat([s_feats, q_feats[hc]], dim=0),
                    torch.cat([s_labels, pl[hc]], dim=0),
                    n_way, fallback=proto)
        P_ctpr = proto
        ctpr_scores = score_with_prototypes(q_feats, P_ctpr, metric=metric)

        if method == 'tcr_ctpr_ensemble':
            # TCR with DEFAULT support init, then ensemble with CTPR scores.
            tcr_logits = tcr_refine(
                s_feats, s_labels, q_feats, n_way=n_way,
                n_iter=args.tcr_iter, lr=args.tcr_lr,
                alpha=args.tcr_alpha, beta=args.tcr_beta,
                temp=args.tcr_temp, metric=args.tcr_metric)
            tn = (tcr_logits - tcr_logits.mean()) / (tcr_logits.std() + 1e-6)
            cn = (ctpr_scores - ctpr_scores.mean()) / (ctpr_scores.std() + 1e-6)
            return args.tcr_ctpr_w * tn + (1 - args.tcr_ctpr_w) * cn

        # trace / trace_ens: TCR with (possibly blended) init.
        if args.ctpr_init_w < 1.0:
            P_sup = compute_prototypes(s_feats, s_labels, n_way)
            P_init = args.ctpr_init_w * P_ctpr + (1 - args.ctpr_init_w) * P_sup
        else:
            P_init = P_ctpr
        tcr_logits = tcr_refine(
            s_feats, s_labels, q_feats, n_way=n_way,
            n_iter=args.tcr_iter, lr=args.tcr_lr,
            alpha=args.tcr_alpha, beta=args.tcr_beta,
            temp=args.tcr_temp, metric=args.tcr_metric,
            init_W=P_init)

        if method == 'trace':
            return tcr_logits

        # trace_ens: ensemble the blend-init TCR output with CTPR scores
        # (mirrors meta_deepbdc.set_forward_with_trace, combine_ctpr=True).
        tn = (tcr_logits - tcr_logits.mean()) / (tcr_logits.std() + 1e-6)
        cn = (ctpr_scores - ctpr_scores.mean()) / (ctpr_scores.std() + 1e-6)
        return args.tcr_ctpr_w * tn + (1 - args.tcr_ctpr_w) * cn

    raise ValueError(f'unknown method {method!r}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True, help='path to cached features .pt')
    p.add_argument('--method', required=True,
                    choices=['plain', 'ctpr', 'tcr', 'tcr_ctpr_ensemble',
                             'trace', 'trace_ens'])
    p.add_argument('--max_tasks', default=0, type=int,
                    help='if >0, only evaluate the first K tasks (for faster '
                         'grid search; 0 = all tasks)')
    p.add_argument('--ctpr_threshold', default=0.5, type=float)
    p.add_argument('--ctpr_iter', default=3, type=int)
    p.add_argument('--tcr_iter', default=100, type=int)
    p.add_argument('--tcr_lr', default=1e-3, type=float)
    p.add_argument('--tcr_alpha', default=0.1, type=float)
    p.add_argument('--tcr_beta', default=1.0, type=float)
    p.add_argument('--tcr_temp', default=15.0, type=float)
    p.add_argument('--tcr_metric', default='cos', choices=['cos', 'dot'])
    p.add_argument('--tcr_ctpr_w', default=0.5, type=float)
    p.add_argument('--ctpr_init_w', default=1.0, type=float)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out_json', default='', help='optional JSON file to append result')
    args = p.parse_args()

    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    sup_t = cache['support_feats']     # [N_eps, n_way*n_shot, D]
    qry_t = cache['query_feats']       # [N_eps, n_way*n_query, D]
    n_way = cache['n_way']
    n_shot = cache['n_shot']
    n_query = cache['n_query']
    n_eps_per_task = cache['n_episodes']
    n_tasks = cache['n_tasks']
    n_total = sup_t.shape[0]
    assert n_total == n_eps_per_task * n_tasks, \
        f'cache mismatch: {n_total} vs {n_eps_per_task}*{n_tasks}'

    # Per-task accuracies (mirror test.py's 5-task averaging)
    print(f'[FAST] {args.method} on {cache["dataset"]} {n_shot}-shot, '
          f'{n_eps_per_task} ep × {n_tasks} tasks (cache: {args.cache})')
    print(f'[FAST] ckpt md5: {cache.get("ckpt_md5", "?")}')

    y_label = np.repeat(range(n_way), n_query)
    task_means = []
    t0 = time.time()

    eval_tasks = n_tasks if args.max_tasks <= 0 else min(args.max_tasks, n_tasks)
    for task_idx in range(eval_tasks):
        accs = []
        start = task_idx * n_eps_per_task
        end = start + n_eps_per_task
        for ep_idx in tqdm.tqdm(range(start, end), total=n_eps_per_task,
                                  desc=f'task {task_idx+1}/{n_tasks}'):
            scores = run_episode(sup_t[ep_idx], qry_t[ep_idx],
                                  n_way, n_shot, args.method, args,
                                  device=args.device)
            pred = scores.argmax(dim=-1).cpu().numpy()
            accs.append((pred == y_label).mean() * 100)
        m = float(np.mean(accs))
        s = float(np.std(accs))
        task_means.append(m)
        print(f'  task {task_idx+1}: {m:.2f}% +- {s/np.sqrt(n_eps_per_task)*1.96:.2f}%')

    grand = float(np.mean(task_means))
    elapsed = time.time() - t0
    print(f'[FAST] {n_tasks} test mean acc = {grand:.2f}%   '
          f'(elapsed {elapsed/60:.1f} min, {elapsed/n_total*1000:.1f} ms/ep)')

    if args.out_json:
        rec = {
            'cache': args.cache, 'method': args.method,
            'dataset': cache['dataset'], 'n_shot': n_shot,
            'grand_acc': grand, 'task_accs': task_means,
            'elapsed_sec': elapsed,
            'args': {
                'ctpr_threshold': args.ctpr_threshold,
                'ctpr_iter': args.ctpr_iter,
                'tcr_iter': args.tcr_iter,
                'tcr_alpha': args.tcr_alpha,
                'tcr_beta': args.tcr_beta,
                'tcr_temp': args.tcr_temp,
                'tcr_metric': args.tcr_metric,
                'tcr_ctpr_w': args.tcr_ctpr_w,
                'ctpr_init_w': args.ctpr_init_w,
            },
        }
        # Append-only JSONL
        with open(args.out_json, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f'[FAST] appended record to {args.out_json}')


if __name__ == '__main__':
    main()
