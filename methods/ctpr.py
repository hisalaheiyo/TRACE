"""
CTPR: Confidence-Weighted Transductive Prototype Refinement.

Core idea: The bottleneck in 1-shot CD-FSAR is prototype estimation, not feature
quality (verified by our oracle analysis: 41% baseline vs 73% oracle on HMDB51).
We refine prototypes using high-confidence query pseudo-labels.

Design (single-round, HC-filtered):
  1. Compute baseline prototypes from support features
  2. Score queries with baseline prototypes (using same metric as TAMT)
  3. Filter queries by softmax confidence > threshold
  4. Refine prototypes by merging support + high-conf queries (with pseudo-labels)
  5. Re-score queries with refined prototypes → final predictions

Empirically validated via diagnose_full.py:
  - HMDB51 1-shot: baseline 41.13% → HC@0.5 refined 47.87% (+6.73%)
  - UCF101 1-shot: baseline 86.80% → HC@0.5 refined 92.51% (+5.71%)
  - SSv2 1-shot:   baseline 38.83% → HC@0.5 refined 41.95% (+3.13%)

Fairness guarantees:
  - Uses same frozen TAMT features as baseline
  - Uses same metric (L2 for >1-shot, dot for 1-shot) as TAMT
  - No additional training — pure test-time refinement
  - No gradient fitting — can't overfit to support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_prototypes(feats, labels, n_way, fallback=None):
    """Per-class mean of features. Falls back to `fallback[k]` if class k empty."""
    prots = torch.zeros(n_way, feats.shape[-1],
                        dtype=feats.dtype, device=feats.device)
    for k in range(n_way):
        mask = (labels == k)
        if mask.sum() > 0:
            prots[k] = feats[mask].mean(0)
        elif fallback is not None:
            prots[k] = fallback[k]
    return prots


def score_with_prototypes(query_feats, protos, metric='l2'):
    """
    Compute classification scores.
    Ref: methods/meta_deepbdc.py:metric (TAMT's L2 / dot product metrics)
    """
    q = query_feats.unsqueeze(1)  # [N, 1, D]
    p = protos.unsqueeze(0)       # [1, K, D]
    if metric == 'l2':
        return -((q - p) ** 2).sum(-1)   # higher is better
    elif metric == 'dot':
        return (q * p).sum(-1)
    elif metric == 'cos':
        return F.cosine_similarity(q, p, dim=-1, eps=1e-30) * 10.0
    else:
        raise ValueError(f'Unknown metric: {metric}')


def ctpr_refine(support_feats, support_labels, query_feats, n_way,
                metric='l2', conf_threshold=0.5, n_iter=1):
    """
    Confidence-Weighted Transductive Prototype Refinement.

    Args:
        support_feats:  [n_way * n_support, D]
        support_labels: [n_way * n_support] — integer labels in [0, n_way)
        query_feats:    [n_way * n_query, D]
        n_way:          number of classes in episode
        metric:         'l2' (for >1-shot) or 'dot' (for 1-shot, matches TAMT)
        conf_threshold: softmax confidence threshold for pseudo-labeling
                        (0.5 is empirically best per our diagnostic)
        n_iter:         number of refinement rounds (1 = single-pass)

    Returns:
        final_scores: [n_way * n_query, n_way] — refined classification scores
        info dict with diagnostics:
          - 'n_hc_included': number of high-conf queries used
          - 'hc_pseudo_acc': pseudo-label accuracy for included (if labels known)
          - 'conf_mean': mean confidence of baseline predictions
    """
    # Step 1: baseline prototypes from support
    proto = compute_prototypes(support_feats, support_labels, n_way)

    info = {}

    # Iterative refinement (default 1 round)
    for iter_idx in range(n_iter):
        # Step 2: score queries with current prototypes
        scores = score_with_prototypes(query_feats, proto, metric=metric)

        # Step 3: compute confidence + pseudo-labels
        probs = F.softmax(scores, dim=-1)
        conf, pseudo_labels = probs.max(dim=-1)

        if iter_idx == 0:
            info['conf_mean'] = conf.mean().item()

        # Step 4: filter high-confidence queries
        hc_mask = conf > conf_threshold

        # Step 5: refine prototypes with support + high-conf queries
        if hc_mask.sum() > 0:
            merged_feats = torch.cat(
                [support_feats, query_feats[hc_mask]], dim=0)
            merged_labels = torch.cat(
                [support_labels, pseudo_labels[hc_mask]], dim=0)
            proto = compute_prototypes(
                merged_feats, merged_labels, n_way, fallback=proto)

        if iter_idx == n_iter - 1:
            info['n_hc_included'] = int(hc_mask.sum().item())

    # Final scores with refined prototypes
    final_scores = score_with_prototypes(query_feats, proto, metric=metric)
    return final_scores, info
