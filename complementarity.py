"""
Error-correlation diagnostic: do CTPR and TCR make the SAME mistakes?

If their errors are highly correlated, fusing them cannot fix the shared
errors -> fusion is fundamentally limited (pivot to independent signals).
If errors are weakly correlated, there is room for a smarter fusion.

For each episode we compute, per query, whether CTPR / TCR are correct, then
aggregate over the dataset:
  - acc_ctpr, acc_tcr                 : individual accuracies
  - p_both_correct                    : both right
  - p_both_wrong                      : both wrong
  - p_ctpr_only / p_tcr_only          : exactly one right (complementary mass)
  - phi                               : phi-coefficient of the two correctness
                                        bernoullis (error correlation)
  - oracle_union (upper bound of any  : fraction where AT LEAST ONE is right
    selector that picks the correct     -> the ceiling a *perfect* per-query
    method per query)                    router between CTPR and TCR could reach
  - disagree_rate                     : fraction where CTPR_pred != TCR_pred
  - acc_on_disagree_ctpr/tcr          : who is right when they disagree

Uses cached features. CTPR and TCR use the paper's hyperparameters.

Usage:
  python complementarity.py --cache cached_feats/HMDB51_1shot.pt --max_episodes 2000
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

from methods.ctpr import compute_prototypes, score_with_prototypes
from methods.tcr import tcr_refine


def ctpr_pred(s_feats, s_labels, q_feats, n_way, metric,
              conf_threshold=0.5, n_iter=3):
    proto = compute_prototypes(s_feats, s_labels, n_way)
    for _ in range(n_iter):
        sc = score_with_prototypes(q_feats, proto, metric=metric)
        conf, pl = F.softmax(sc, dim=-1).max(dim=-1)
        hc = conf > conf_threshold
        if hc.sum() > 0:
            proto = compute_prototypes(
                torch.cat([s_feats, q_feats[hc]], 0),
                torch.cat([s_labels, pl[hc]], 0),
                n_way, fallback=proto)
    return score_with_prototypes(q_feats, proto, metric=metric).argmax(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--max_episodes', default=2000, type=int)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    sup = cache['support_feats']; qry = cache['query_feats']
    n_way = cache['n_way']; n_shot = cache['n_shot']; n_query = cache['n_query']
    metric = 'dot' if n_shot == 1 else 'l2'
    N = min(args.max_episodes, sup.shape[0]) if args.max_episodes else sup.shape[0]
    dev = args.device

    q_labels = torch.arange(n_way, device=dev).repeat_interleave(n_query)
    s_labels = torch.arange(n_way, device=dev).repeat_interleave(n_shot)

    both_c = both_w = ctpr_only = tcr_only = 0
    disagree = 0
    dis_ctpr_right = dis_tcr_right = 0
    n_q_total = 0
    ctpr_correct_tot = tcr_correct_tot = 0

    for ep in range(N):
        s = sup[ep].to(dev); q = qry[ep].to(dev)
        with torch.no_grad():
            cp = ctpr_pred(s, s_labels, q, n_way, metric)
        tcr_logits_full = tcr_refine(s, s_labels, q, n_way=n_way, n_iter=100, lr=1e-3,
                        alpha=0.1, beta=1.0, temp=15.0, metric='cos')
        tp = tcr_logits_full.argmax(-1)

        c_ok = (cp == q_labels)
        t_ok = (tp == q_labels)
        both_c += (c_ok & t_ok).sum().item()
        both_w += ((~c_ok) & (~t_ok)).sum().item()
        ctpr_only += (c_ok & (~t_ok)).sum().item()
        tcr_only += ((~c_ok) & t_ok).sum().item()
        dis = (cp != tp)
        disagree += dis.sum().item()
        dis_ctpr_right += (dis & c_ok).sum().item()
        dis_tcr_right += (dis & t_ok).sum().item()
        ctpr_correct_tot += c_ok.sum().item()
        tcr_correct_tot += t_ok.sum().item()
        n_q_total += q_labels.numel()

    n = n_q_total
    acc_ctpr = ctpr_correct_tot / n
    acc_tcr = tcr_correct_tot / n
    p_bc = both_c / n; p_bw = both_w / n
    p_co = ctpr_only / n; p_tcr_only = tcr_only / n
    # phi coefficient between the two correctness indicators
    # cells: a=both correct, b=ctpr correct/tcr wrong, c=ctpr wrong/tcr correct, d=both wrong
    a, b, c, d = both_c, ctpr_only, tcr_only, both_w
    denom = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    phi = (a*d - b*c) / denom if denom > 0 else 0.0
    oracle_union = (both_c + ctpr_only + tcr_only) / n   # at-least-one-right
    disagree_rate = disagree / n
    dis_ctpr_acc = dis_ctpr_right / disagree if disagree else 0
    dis_tcr_acc = dis_tcr_right / disagree if disagree else 0

    print(f"\n===== Error-correlation: {cache['dataset']} {n_shot}-shot "
          f"({N} ep) =====")
    print(f"acc_CTPR              {100*acc_ctpr:.2f}")
    print(f"acc_TCR               {100*acc_tcr:.2f}")
    print(f"both correct          {100*p_bc:.2f}")
    print(f"both wrong            {100*p_bw:.2f}")
    print(f"CTPR-only correct     {100*p_co:.2f}")
    print(f"TCR-only correct      {100*p_tcr_only:.2f}")
    print(f"complementary mass    {100*(p_co+p_tcr_only):.2f}  (exactly one right)")
    print(f"phi (error corr)      {phi:.3f}   [1=identical, 0=independent]")
    print(f"oracle-union ceiling  {100*oracle_union:.2f}  (perfect router)")
    print(f"disagree rate         {100*disagree_rate:.2f}")
    print(f"  when disagree: CTPR right {100*dis_ctpr_acc:.1f} | TCR right {100*dis_tcr_acc:.1f}")
    # interpretation hint
    indep_phi = acc_ctpr*acc_tcr  # not used; phi is the metric
    print(f"\nINTERPRETATION:")
    print(f"  oracle-union ({100*oracle_union:.1f}) vs best single "
          f"({100*max(acc_ctpr,acc_tcr):.1f}) = "
          f"+{100*(oracle_union-max(acc_ctpr,acc_tcr)):.1f} "
          f"-> max possible gain from PERFECT CTPR/TCR routing")

    out = f"logs/errcorr_{cache['dataset']}_{n_shot}shot.json"
    json.dump({'dataset': cache['dataset'], 'n_shot': n_shot, 'N': N,
               'acc_ctpr': acc_ctpr, 'acc_tcr': acc_tcr,
               'both_correct': p_bc, 'both_wrong': p_bw,
               'ctpr_only': p_co, 'tcr_only': p_tcr_only,
               'phi': phi, 'oracle_union': oracle_union,
               'disagree_rate': disagree_rate,
               'dis_ctpr_acc': dis_ctpr_acc, 'dis_tcr_acc': dis_tcr_acc},
              open(out, 'w'), indent=2)
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
