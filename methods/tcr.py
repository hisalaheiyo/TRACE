"""
TCR: Transductive Information Maximization for few-shot learning.
Ref: Boudiaf et al., "Information Maximization for Few-Shot Learning", NeurIPS 2020.

A test-time method fundamentally different from CTPR:
  - CTPR    : pseudo-label high-confidence queries, refine prototype (greedy, 1-pass).
  - TCR     : optimize a tiny per-episode classifier W via an information-theoretic
              objective over the WHOLE query set (joint, iterative).

Per episode, minimize over W [n_way, D] (init from support prototypes):
    L(W) = CE(support)              # fit the few labeled supports
         + alpha * H(Y_q | X_q)     # query predictions should be confident
         - beta  * H(Y_q)           # query predictions should be class-balanced

Only W is optimized; backbone features are frozen. Fully test-time.
"""

import torch
import torch.nn.functional as F
from .ctpr import compute_prototypes


def tcr_refine(support_feats, support_labels, query_feats, n_way,
               n_iter=100, lr=1e-3, alpha=0.1, beta=0.1, temp=15.0,
               metric='cos', init_W=None):
    """
    Args:
        support_feats:  [N_s, D]
        support_labels: [N_s]  ints in [0, n_way)
        query_feats:    [N_q, D]
        n_iter:  gradient steps on W (0 = plain prototype classification)
        lr:      learning rate for W
        alpha:   weight of conditional entropy  H(Y_q|X_q)  (>0, minimize)
        beta:    weight of marginal entropy     H(Y_q)      (>0, maximize → subtract)
        temp:    logit temperature (for 'cos' metric)
        metric:  'cos' (cosine, TCR standard) or 'dot'
        init_W:  [n_way, D] optional initialization for W. If None (default),
                 init from support-only prototypes (TCR standard). When passed
                 a CTPR-refined prototype, this implements the
                 "CTPR-init TCR" fusion (TRACE).
                 IMPORTANT: any tensor passed here will be .detach().clone()'d
                 — the caller's tensor is NOT modified, and TCR still owns
                 its own optimizable W.
    Returns:
        query_logits: [N_q, n_way]
    """
    # Init classifier weights. Default = support-only prototypes (TCR standard).
    # Override = CTPR-refined prototype (TRACE: transductive-informed init).
    if init_W is None:
        W = compute_prototypes(support_feats, support_labels, n_way).detach().clone()
    else:
        assert init_W.shape == (n_way, support_feats.shape[-1]), \
            f'init_W shape {tuple(init_W.shape)} must be (n_way, D)'
        W = init_W.detach().clone().to(support_feats.device)
    W.requires_grad_(True)
    support_oh = F.one_hot(support_labels, n_way).float()   # [N_s, n_way]

    def logits(feat, w):
        if metric == 'cos':
            return temp * (F.normalize(feat, dim=-1) @ F.normalize(w, dim=-1).t())
        elif metric == 'dot':
            return feat @ w.t()
        else:
            raise ValueError(f'unknown metric {metric!r}')

    if n_iter > 0:
        opt = torch.optim.Adam([W], lr=lr)
        for _ in range(n_iter):
            s_logit = logits(support_feats, W)
            q_logit = logits(query_feats, W)

            # support cross-entropy
            ce = -(support_oh * F.log_softmax(s_logit, dim=-1)).sum(-1).mean()

            # query probabilities
            pq = F.softmax(q_logit, dim=-1)                      # [N_q, n_way]
            log_pq = F.log_softmax(q_logit, dim=-1)

            # conditional entropy H(Y|X): per-sample entropy, averaged → minimize
            h_cond = -(pq * log_pq).sum(-1).mean()

            # marginal entropy H(Y): entropy of mean prediction → maximize
            p_marg = pq.mean(0)                                  # [n_way]
            h_marg = -(p_marg * (p_marg + 1e-12).log()).sum()

            loss = ce + alpha * h_cond - beta * h_marg
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        return logits(query_feats, W).detach()
