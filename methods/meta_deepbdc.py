import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from .template import MetaTemplate
from .bdc_module import BDC
from sklearn.linear_model import LogisticRegression

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.class_way='add'
        self.method = params.method
        # self.class_way='score'

    def set_forward(self, x, is_feature=False):
        if self.class_way=='add':
            z_support, z_query, support_feature, query_feature = self.parse_feature(x, is_feature)#x torch.Size([5, 17, 3, 16,224, 224])

            z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)

            z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

            scores1 = self.metric(z_query, z_proto)

            return scores1
            # scores = self.euclidean_dist(z_query, z_proto)

            # frame_sim = cos_sim(query_feature,support_feature)
            # frame_dists = 1 - frame_sim

            # dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = self.n_query*self.n_way, sb = self.n_support*self.n_way)  # [50, 5, 8, 8]
            # cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))  #去掉试试？？
            # scores2 = -cum_dists
            # return scores1 + scores2
        
        if self.class_way=='score':
            z_support1, z_query1,z_support2, z_query2 = self.parse_feature(x, is_feature)
            z_proto1 = z_support1.contiguous().view(self.n_way, self.n_support, -1).mean(1)
            z_query1 = z_query1.contiguous().view(self.n_way * self.n_query, -1)
            scores1 = self.metric(z_query1, z_proto1)
            z_proto2 = z_support2.contiguous().view(self.n_way, self.n_support, -1).mean(1)
            z_query2 = z_query2.contiguous().view(self.n_way * self.n_query, -1)
            scores2 = self.metric(z_query2, z_proto2)
            scores = scores1 + scores2
            return scores

    def _extract_ep_feats(self, x):
        """Shared feature extraction helper (used by CTPR variants)."""
        with torch.no_grad():
            z_s, z_q, _, _ = self.parse_feature(x, is_feature=False)
            sup = z_s.contiguous().view(self.n_way * self.n_support, -1)
            qry = z_q.contiguous().view(self.n_way * self.n_query, -1)
        sup_lbl = torch.arange(self.n_way, device=sup.device).repeat_interleave(self.n_support)
        return sup, sup_lbl, qry

    def set_forward_with_ctpr_iterative(self, x, conf_threshold=0.5, n_iter=3):
        """V1: Iterative CTPR (multi-round)."""
        from .ctpr_enhanced import ctpr_iterative
        sup, sup_lbl, qry = self._extract_ep_feats(x)
        metric = 'dot' if self.n_support == 1 else 'l2'
        with torch.no_grad():
            return ctpr_iterative(sup, sup_lbl, qry, n_way=self.n_way,
                                    metric=metric, conf_threshold=conf_threshold,
                                    n_iter=n_iter)

    def set_forward_with_ctpr_soft(self, x):
        """V2: Soft CTPR (EM-style soft assignment)."""
        from .ctpr_enhanced import ctpr_soft
        sup, sup_lbl, qry = self._extract_ep_feats(x)
        metric = 'dot' if self.n_support == 1 else 'l2'
        with torch.no_grad():
            return ctpr_soft(sup, sup_lbl, qry, n_way=self.n_way, metric=metric)

    def set_forward_with_ctpr_balanced(self, x, conf_threshold=0.5, top_m_per_class=None):
        """V3: Class-balanced HC filtering."""
        from .ctpr_enhanced import ctpr_class_balanced
        sup, sup_lbl, qry = self._extract_ep_feats(x)
        metric = 'dot' if self.n_support == 1 else 'l2'
        with torch.no_grad():
            return ctpr_class_balanced(sup, sup_lbl, qry, n_way=self.n_way,
                                        metric=metric, conf_threshold=conf_threshold,
                                        top_m_per_class=top_m_per_class)

    def set_forward_with_gctpr(self, x, conf_threshold=0.5, n_iter=1,
                               k_neighbors=5, lp_alpha=0.5, lp_iter=5,
                               require_agreement=True, query_init='softmax'):
        """GCTPR: graph-based HC filtering via label propagation."""
        from .gctpr import gctpr_refine
        sup, sup_lbl, qry = self._extract_ep_feats(x)
        metric = 'dot' if self.n_support == 1 else 'l2'
        with torch.no_grad():
            scores, info = gctpr_refine(
                sup, sup_lbl, qry, n_way=self.n_way, metric=metric,
                k_neighbors=k_neighbors, lp_alpha=lp_alpha, lp_iter=lp_iter,
                conf_threshold=conf_threshold, n_iter=n_iter,
                require_agreement=require_agreement, query_init=query_init)
        # Diagnostic: print once
        if not getattr(self, '_gctpr_diag_printed', False):
            print(f'[GCTPR-diag] prop_conf max: mean={info.get("prop_conf_max_mean", 0):.4f} '
                  f'max={info.get("prop_conf_max_max", 0):.4f} '
                  f'n_hc={info.get("n_hc", 0)}/{qry.shape[0]} at thresh={conf_threshold}')
            self._gctpr_diag_printed = True
        return scores

    def set_forward_with_tempctpr(self, x, conf_threshold=0.5, n_iter=1,
                                  js_threshold=0.0, combine='and'):
        """TempCTPR: temporal-reverse consistency HC filter (with one-time diag)."""
        from .tempctpr import tempctpr_refine

        with torch.no_grad():
            z_s, z_q, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_s.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_q.contiguous().view(self.n_way * self.n_query, -1)

        x_rev = torch.flip(x, dims=[3])
        with torch.no_grad():
            _, z_q_rev, _, _ = self.parse_feature(x_rev, is_feature=False)
            query_feats_rev = z_q_rev.contiguous().view(self.n_way * self.n_query, -1)

        support_labels = torch.arange(self.n_way, device=support_feats.device
                                      ).repeat_interleave(self.n_support)
        metric = 'dot' if self.n_support == 1 else 'l2'

        with torch.no_grad():
            scores, info = tempctpr_refine(
                support_feats, support_labels, query_feats, query_feats_rev,
                n_way=self.n_way, metric=metric,
                conf_threshold=conf_threshold, n_iter=n_iter,
                js_threshold=js_threshold, combine=combine)
        if not getattr(self, '_tempctpr_diag_printed', False):
            print(f'[TempCTPR-diag] js_mean={info.get("js_mean", 0):.4f} '
                  f'n_hc={info.get("n_hc", 0)}/{query_feats.shape[0]} '
                  f'combine={combine} conf_thr={conf_threshold} js_thr={js_threshold}')
            self._tempctpr_diag_printed = True
        return scores

    def _extract_support_feats_aug(self, x_support):
        """Support-only feature extraction via parse_feature, temporarily setting n_query=0."""
        saved_n_query = self.n_query
        self.n_query = 0
        try:
            z_sup, _, _, _ = self.parse_feature(x_support, is_feature=False)
        finally:
            self.n_query = saved_n_query
        return z_sup  # [n_way, n_support, 384]

    def set_forward_with_ctpr_sta(self, x, K=4, conf_threshold=0.5, n_iter=1,
                                  use_ctpr=True):
        """Support-TTA + CTPR:
          1. View 0: forward full episode (original) → z_sup_0, z_query
          2. Views 1..K-1: forward augmented support only → z_sup_k
          3. Average support features across K views → reduces 1-shot sampling noise
          4. Run standard CTPR refinement on (averaged support, original query)

        Args:
          K: number of augmented views of support (including original). K=1 = no TTA.
          use_ctpr: if False, just do plain classification with averaged support.
        """
        from .ctpr import ctpr_refine
        from .support_tta import augment_support

        # View 0: full episode, get both support and query features
        with torch.no_grad():
            z_sup_0, z_qry, _, _ = self.parse_feature(x, is_feature=False)

        sup_feats_sum = z_sup_0.clone()

        # Views 1..K-1: support only with augmentation
        x_support = x[:, :self.n_support]  # [n_way, n_support, C, T, H, W]
        with torch.no_grad():
            for k in range(1, K):
                x_sup_aug = augment_support(x_support, k)
                z_sup_k = self._extract_support_feats_aug(x_sup_aug)
                sup_feats_sum = sup_feats_sum + z_sup_k

        z_support_avg = sup_feats_sum / K   # [n_way, n_support, 384]

        # Flatten
        support_feats = z_support_avg.contiguous().view(self.n_way * self.n_support, -1)
        query_feats = z_qry.contiguous().view(self.n_way * self.n_query, -1)

        metric = 'dot' if self.n_support == 1 else 'l2'

        if not use_ctpr:
            # Plain classification with TTA-averaged prototype
            proto = z_support_avg.mean(1)  # [n_way, 384]
            return self.metric(query_feats, proto)

        # CTPR refine on top of TTA-averaged support
        support_labels = torch.arange(self.n_way, device=support_feats.device
                                      ).repeat_interleave(self.n_support)
        with torch.no_grad():
            scores, _ = ctpr_refine(
                support_feats, support_labels, query_feats,
                n_way=self.n_way, metric=metric,
                conf_threshold=conf_threshold, n_iter=n_iter)
        return scores

    def set_forward_with_rpa(self, x, base_features, K=5, alpha=0.3,
                             mode='blend', use_ctpr=True, n_iter=1,
                             conf_threshold=0.5, metric='cosine'):
        """Retrieval-based Prototype Augmentation.

        Args:
            base_features: dict with keys 'features' [N_base, D], 'labels' [N_base]
            K: retrieval neighbors per support
            alpha: blend weight (only used in mode='blend')
            mode: 'blend' (weighted average) or 'expand' (add as new support samples)
            use_ctpr: combine with CTPR after retrieval augmentation
            n_iter: CTPR outer iterations (1=hard, 3=iter3) if use_ctpr
            metric: 'cosine' or 'l2' for retrieval
        """
        from .retrieval_aug import rpa_augment_support, rpa_expand_support
        from .ctpr import ctpr_refine, compute_prototypes, score_with_prototypes

        with torch.no_grad():
            z_sup, z_qry, _, _ = self.parse_feature(x, is_feature=False)
        support_feats = z_sup.contiguous().view(self.n_way * self.n_support, -1)
        query_feats = z_qry.contiguous().view(self.n_way * self.n_query, -1)

        support_labels = torch.arange(self.n_way, device=support_feats.device
                                      ).repeat_interleave(self.n_support)

        base_feats_gpu = base_features['features'].to(support_feats.device)

        with torch.no_grad():
            if mode == 'blend':
                enhanced_sup, enhanced_lbl, info = rpa_augment_support(
                    support_feats, support_labels, base_feats_gpu,
                    n_way=self.n_way, K=K, alpha=alpha, metric=metric)
            elif mode == 'expand':
                enhanced_sup, enhanced_lbl, info = rpa_expand_support(
                    support_feats, support_labels, base_feats_gpu,
                    n_way=self.n_way, K=K, metric=metric)
            else:
                raise ValueError(mode)

        if not getattr(self, '_rpa_diag_printed', False):
            print(f'[RPA-diag] mode={mode} K={K} alpha={alpha} metric={metric} '
                  f'mean_sim={info["mean_sim"]:.4f} enhanced_sup={tuple(enhanced_sup.shape)}')
            self._rpa_diag_printed = True

        score_metric = 'dot' if self.n_support == 1 else 'l2'

        with torch.no_grad():
            if use_ctpr:
                scores, _ = ctpr_refine(
                    enhanced_sup, enhanced_lbl, query_feats,
                    n_way=self.n_way, metric=score_metric,
                    conf_threshold=conf_threshold, n_iter=n_iter)
            else:
                proto = compute_prototypes(enhanced_sup, enhanced_lbl, self.n_way)
                scores = score_with_prototypes(query_feats, proto, metric=score_metric)
        return scores

    def set_forward_with_badc(self, x, base_stats, k_neighbors=3, n_virtual=100, alpha=0.21):
        """BADC only: use base class stats to calibrate novel prototypes."""
        from .badc import badc_refine
        with torch.no_grad():
            z_support, z_query, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_support.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_query.contiguous().view(self.n_way * self.n_query, -1)
        support_labels = torch.arange(self.n_way, device=support_feats.device).repeat_interleave(self.n_support)
        metric = 'dot' if self.n_support == 1 else 'l2'
        with torch.no_grad():
            scores, _ = badc_refine(
                support_feats, support_labels, query_feats,
                n_way=self.n_way, base_stats=base_stats,
                k_neighbors=k_neighbors, n_virtual=n_virtual, alpha=alpha,
                metric=metric)
        return scores

    def set_forward_with_dspr(self, x, base_stats, conf_threshold=0.5,
                               k_neighbors=3, n_virtual=100, alpha=0.21):
        """DSPR: combine CTPR + BADC."""
        from .badc import dspr_refine
        with torch.no_grad():
            z_support, z_query, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_support.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_query.contiguous().view(self.n_way * self.n_query, -1)
        support_labels = torch.arange(self.n_way, device=support_feats.device).repeat_interleave(self.n_support)
        metric = 'dot' if self.n_support == 1 else 'l2'
        with torch.no_grad():
            scores = dspr_refine(
                support_feats, support_labels, query_feats,
                n_way=self.n_way, base_stats=base_stats,
                ctpr_threshold=conf_threshold, k_neighbors=k_neighbors,
                n_virtual=n_virtual, alpha=alpha, metric=metric)
        return scores

    def set_forward_with_ctpr(self, x, conf_threshold=0.5, n_iter=1):
        """
        CTPR: Confidence-Weighted Transductive Prototype Refinement.

        Training-free test-time refinement. Uses TAMT's frozen pipeline +
        same metric as baseline. Improvement comes from using high-confidence
        query pseudo-labels to refine prototypes.

        Fairness: Same features, same metric, same episodes as baseline.
        Novelty: Utilizes query set information that baseline ignores.
        """
        from .ctpr import ctpr_refine

        # Step 1: Extract features through frozen pipeline (no gradient)
        with torch.no_grad():
            z_support, z_query, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_support.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Step 2: Build support labels matching TAMT's episode layout
        support_labels = torch.arange(self.n_way, device=support_feats.device)
        support_labels = support_labels.repeat_interleave(self.n_support)

        # Step 3: Determine metric — MUST match TAMT's self.metric() behavior:
        #   - 1-shot: dot product (line 150-151 of this file)
        #   - >1-shot: L2 distance (line 148-149)
        metric = 'dot' if self.n_support == 1 else 'l2'

        # Step 4: Run CTPR refinement
        with torch.no_grad():
            scores, info = ctpr_refine(
                support_feats, support_labels, query_feats,
                n_way=self.n_way, metric=metric,
                conf_threshold=conf_threshold, n_iter=n_iter,
            )
        return scores

    def set_forward_with_ctpr_combo(self, x, conf_threshold=0.5, n_iter=3,
                                    nsae_weights=None, use_standardize=False):
        """Combo-1: CTPR with optional test-time feature preprocessing.
          - NSAE channel re-weighting (suppress noisy channels)
          - Episode feature standardization (reduce cross-domain shift)
        Both are applied to features BEFORE prototype computation; all three
        components (NSAE, Std, CTPR) are orthogonal test-time operations.
        """
        from .ctpr import ctpr_refine
        from .nsae import apply_nsae, standardize_episode

        with torch.no_grad():
            z_support, z_query, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_support.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # --- NSAE channel weighting ---
        if nsae_weights is not None:
            w = nsae_weights.to(support_feats.device)
            support_feats = apply_nsae(support_feats, w)
            query_feats = apply_nsae(query_feats, w)

        # --- Episode standardization ---
        if use_standardize:
            support_feats, query_feats = standardize_episode(support_feats, query_feats)

        support_labels = torch.arange(self.n_way, device=support_feats.device)
        support_labels = support_labels.repeat_interleave(self.n_support)
        metric = 'dot' if self.n_support == 1 else 'l2'

        with torch.no_grad():
            scores, _ = ctpr_refine(
                support_feats, support_labels, query_feats,
                n_way=self.n_way, metric=metric,
                conf_threshold=conf_threshold, n_iter=n_iter,
            )
        return scores

    def set_forward_with_trx(self, x, temperature=0.1, use_card2=True,
                             combine_ctpr=False, conf_threshold=0.5, n_iter=3):
        """TRX option-A: parameter-free frame-level cross-attention matching.

        Operates on backbone frame tokens [T, D] (NOT the pooled BDC feature).
        combine_ctpr: if True, ensemble TRX scores with CTPR scores 50/50.
        """
        from .trx_match import trx_match

        with torch.no_grad():
            # frame tokens [n_way*n_sample, T, C]
            tokens = self._backbone_tokens(x)
            T, C = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(self.n_way, self.n_support + self.n_query, T, C)
            support_tok = tokens[:, :self.n_support].reshape(
                self.n_way * self.n_support, T, C)
            query_tok = tokens[:, self.n_support:].reshape(
                self.n_way * self.n_query, T, C)

        support_labels = torch.arange(self.n_way, device=support_tok.device
                                      ).repeat_interleave(self.n_support)
        with torch.no_grad():
            trx_scores = trx_match(query_tok, support_tok, support_labels,
                                   n_way=self.n_way, n_support=self.n_support,
                                   temperature=temperature, use_card2=use_card2)

        if not combine_ctpr:
            return trx_scores

        # Ensemble with CTPR (on pooled BDC features)
        with torch.no_grad():
            from .ctpr import ctpr_refine
            z_s, z_q, _, _ = self.parse_feature(x, is_feature=False)
            sup = z_s.contiguous().view(self.n_way * self.n_support, -1)
            qry = z_q.contiguous().view(self.n_way * self.n_query, -1)
            metric = 'dot' if self.n_support == 1 else 'l2'
            ctpr_scores, _ = ctpr_refine(sup, support_labels, qry,
                                         n_way=self.n_way, metric=metric,
                                         conf_threshold=conf_threshold, n_iter=n_iter)
        # normalize each to comparable scale, then average
        trx_n = (trx_scores - trx_scores.mean()) / (trx_scores.std() + 1e-6)
        ctpr_n = (ctpr_scores - ctpr_scores.mean()) / (ctpr_scores.std() + 1e-6)
        return 0.5 * trx_n + 0.5 * ctpr_n

    def set_forward_with_tcr(self, x, n_iter=100, lr=1e-3, alpha=0.1, beta=0.1,
                             temp=15.0, metric='cos', combine_ctpr=False,
                             conf_threshold=0.5, ctpr_n_iter=3,
                             tcr_ctpr_w=0.5):
        """TCR: Transductive Information Maximization (test-time, frozen features).
        combine_ctpr: if True, ensemble normalized TCR + CTPR logits.
        tcr_ctpr_w: weight on TCR in the ensemble; 1-w on CTPR. Default 0.5
            preserves the previously validated 'tcr_ctpr_ensemble' behaviour.
        """
        from .tcr import tcr_refine

        with torch.no_grad():
            z_s, z_q, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_s.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_q.contiguous().view(self.n_way * self.n_query, -1)
        support_labels = torch.arange(self.n_way, device=support_feats.device
                                      ).repeat_interleave(self.n_support)

        tcr_logits = tcr_refine(support_feats, support_labels, query_feats,
                                n_way=self.n_way, n_iter=n_iter, lr=lr,
                                alpha=alpha, beta=beta, temp=temp, metric=metric)
        if not combine_ctpr:
            return tcr_logits

        from .ctpr import ctpr_refine
        m = 'dot' if self.n_support == 1 else 'l2'
        with torch.no_grad():
            ctpr_logits, _ = ctpr_refine(support_feats, support_labels, query_feats,
                                         n_way=self.n_way, metric=m,
                                         conf_threshold=conf_threshold,
                                         n_iter=ctpr_n_iter)
        tcr_n = (tcr_logits - tcr_logits.mean()) / (tcr_logits.std() + 1e-6)
        ctpr_n = (ctpr_logits - ctpr_logits.mean()) / (ctpr_logits.std() + 1e-6)
        return tcr_ctpr_w * tcr_n + (1.0 - tcr_ctpr_w) * ctpr_n

    def set_forward_with_trace(self, x,
                                        conf_threshold=0.5, ctpr_n_iter=3,
                                        tcr_n_iter=100, tcr_lr=1e-3,
                                        tcr_alpha=0.1, tcr_beta=1.0,
                                        tcr_temp=15.0, tcr_metric='cos',
                                        combine_ctpr=False, tcr_ctpr_w=0.5,
                                        ctpr_init_w=1.0):
        """TRACE: CTPR-init TCR (transductive-informed initialization).

        Step 1: run CTPR (multi-round) -> refined prototype P_ctpr
        Step 2: run TCR with W initialized from a blend:
                   W_init = ctpr_init_w * P_ctpr + (1 - ctpr_init_w) * P_support
                ctpr_init_w=1.0 (default) -> pure CTPR init (original TRACE).
                ctpr_init_w=0.0           -> pure support init (= TCR-only).
                Intermediate values blend the two — used to diagnose datasets
                like HMDB 1-shot where pure CTPR init harms TCR convergence
                (see hmdb_diag_17919212.out).
        Step 3: return TCR logits (or normalized ensemble with CTPR if requested)

        Motivation: TCR's info-max objective is non-convex; with 1-shot the
        support-only init is noisy. CTPR-refined prototypes use HC query info,
        giving TCR a denoised starting point. Single forward pass, no
        per-episode hyperparameter tuning beyond the existing CTPR / TCR ones.

        Fairness: frozen TAMT pipeline, identical metric handling as
        existing CTPR (dot/l2 by shot) and TCR (cos by default). Default
        ctpr_init_w=1.0 keeps the previously validated TRACE behaviour
        bit-for-bit identical.
        """
        from .tcr import tcr_refine
        from .ctpr import ctpr_refine, compute_prototypes

        with torch.no_grad():
            z_s, z_q, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_s.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_q.contiguous().view(self.n_way * self.n_query, -1)
        support_labels = torch.arange(self.n_way, device=support_feats.device
                                      ).repeat_interleave(self.n_support)

        # --- Step 1: CTPR refines the prototype using HC pseudo-labels ---
        # Use the TAMT-canonical CTPR metric (dot/l2 by shot) to determine
        # WHICH queries are HC; the resulting prototype is in feature space
        # and is independent of which metric TCR later uses.
        ctpr_metric = 'dot' if self.n_support == 1 else 'l2'
        with torch.no_grad():
            # Re-implement the loop so we can recover the FINAL prototype
            # (ctpr_refine returns scores; we need P itself for TCR init).
            proto = compute_prototypes(support_feats, support_labels, self.n_way)
            for _ in range(ctpr_n_iter):
                from torch.nn.functional import softmax
                from .ctpr import score_with_prototypes
                sc = score_with_prototypes(query_feats, proto, metric=ctpr_metric)
                conf, pl = softmax(sc, dim=-1).max(dim=-1)
                hc = conf > conf_threshold
                if hc.sum() > 0:
                    proto = compute_prototypes(
                        torch.cat([support_feats, query_feats[hc]], dim=0),
                        torch.cat([support_labels, pl[hc]], dim=0),
                        self.n_way, fallback=proto)
            P_ctpr = proto  # [n_way, D]

            # Blend CTPR-refined proto with support proto (TRACE).
            # ctpr_init_w=1.0 → pure CTPR (original TRACE; default).
            # ctpr_init_w=0.0 → pure support proto (equivalent to TCR-only).
            if ctpr_init_w < 1.0:
                P_sup = compute_prototypes(
                    support_feats, support_labels, self.n_way)
                P_init = ctpr_init_w * P_ctpr + (1.0 - ctpr_init_w) * P_sup
            else:
                P_init = P_ctpr

        # --- Step 2: TCR with (possibly blended) refined prototype as init_W ---
        tcr_logits = tcr_refine(
            support_feats, support_labels, query_feats,
            n_way=self.n_way, n_iter=tcr_n_iter, lr=tcr_lr,
            alpha=tcr_alpha, beta=tcr_beta, temp=tcr_temp, metric=tcr_metric,
            init_W=P_init,
        )

        # --- Step 3: optional light ensemble with the same CTPR scores ---
        if not combine_ctpr:
            return tcr_logits

        with torch.no_grad():
            ctpr_logits, _ = ctpr_refine(
                support_feats, support_labels, query_feats,
                n_way=self.n_way, metric=ctpr_metric,
                conf_threshold=conf_threshold, n_iter=ctpr_n_iter,
            )
        tcr_n  = (tcr_logits  - tcr_logits.mean())  / (tcr_logits.std()  + 1e-6)
        ctpr_n = (ctpr_logits - ctpr_logits.mean()) / (ctpr_logits.std() + 1e-6)
        return tcr_ctpr_w * tcr_n + (1.0 - tcr_ctpr_w) * ctpr_n

    def set_forward_with_maf(self, x, conf_threshold=0.5, n_iter=1,
                              tcr_n_iter=100, tcr_lr=1e-3,
                              tcr_alpha=0.1, tcr_beta=1.0,
                              tcr_temp=15.0, tcr_metric='cos',
                              ensemble_w=0.0):
        """MAF: Mutual-Agreement Filtering — CTPR ∩ TCR agreement-based HC mask.
        Single forward; no parameter updates. See methods/maf.py for the algorithm.
        Fairness: same features, same TAMT metric, same TCR hyperparams as
        validated 'tcr_ctpr_ensemble' mode.
        """
        from .maf import maf_refine

        with torch.no_grad():
            z_s, z_q, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_s.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_q.contiguous().view(self.n_way * self.n_query, -1)
        support_labels = torch.arange(self.n_way, device=support_feats.device
                                      ).repeat_interleave(self.n_support)
        # CTPR-side metric MUST mirror TAMT (line 380-383 of this file).
        ctpr_metric = 'dot' if self.n_support == 1 else 'l2'

        scores, _info = maf_refine(
            support_feats, support_labels, query_feats, n_way=self.n_way,
            metric=ctpr_metric, conf_threshold=conf_threshold, n_iter=n_iter,
            tcr_n_iter=tcr_n_iter, tcr_lr=tcr_lr,
            tcr_alpha=tcr_alpha, tcr_beta=tcr_beta,
            tcr_temp=tcr_temp, tcr_metric=tcr_metric,
            ensemble_w=ensemble_w,
        )
        return scores

    def set_forward_with_tsra(self, x, tsra_module, max_iter=40, lr=0.5, distance='cos',
                               lam_entropy=0.0, lam_wd=0.0, early_stop_iter=None):
        """
        Test-time adaptation via TSRA.

        Ref: TSA (CVPR'22) base design
        Ref: TENT (ICLR'21) entropy regularization on query
        """
        from .tsra import fit_tsra

        # Step 1: Extract features (frozen)
        with torch.no_grad():
            z_support, z_query, _, _ = self.parse_feature(x, is_feature=False)
            support_feats = z_support.contiguous().view(self.n_way * self.n_support, -1)
            query_feats = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Step 2: Labels
        support_labels = torch.arange(self.n_way, device=support_feats.device)
        support_labels = support_labels.repeat_interleave(self.n_support)

        # Step 3: Fit TSRA (optionally with query entropy reg)
        fit_tsra(tsra_module, support_feats.detach(), support_labels,
                 n_way=self.n_way, max_iter=max_iter, lr=lr, distance=distance,
                 query_features=query_feats.detach() if lam_entropy > 0 else None,
                 lam_entropy=lam_entropy, lam_wd=lam_wd,
                 early_stop_iter=early_stop_iter)

        # Step 4: Apply adapted TSRA (no_grad for inference)
        with torch.no_grad():
            adapted_support = tsra_module(support_feats)   # [N_s, D]
            adapted_query = tsra_module(query_feats)        # [N_q, D]

            # Compute prototypes from adapted support (as in set_forward)
            z_proto = adapted_support.view(self.n_way, self.n_support, -1).mean(1)
            # Score via same metric as self.metric
            scores = self.metric(adapted_query, z_proto)

        return scores

    def set_forward_tcmt(self, x, tau=1.0, detach_probs=True, warmup=False):
        """
        TCMT forward (Transductive-Consistent Meta-Training).
        Soft-CTPR in the training loop: prototype = support + prob-weighted query.
        Gradients flow through refined proto → TAA+GTMT learn CTPR-friendly features.
        warmup=True degrades to plain support-only prototype (standard forward).
        """
        z_support, z_query, _, _ = self.parse_feature(x, is_feature=False)

        support_feats = z_support.contiguous().view(self.n_way * self.n_support, -1)
        query_feats = z_query.contiguous().view(self.n_way * self.n_query, -1)
        D = support_feats.shape[-1]

        support_labels = torch.arange(self.n_way, device=support_feats.device)
        support_labels = support_labels.repeat_interleave(self.n_support)

        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)

        if warmup:
            return self.metric(query_feats, z_proto)

        scores_init = self.metric(query_feats, z_proto)
        probs = F.softmax(scores_init / tau, dim=-1)
        if detach_probs:
            probs = probs.detach()

        refined_proto = torch.zeros(self.n_way, D, device=support_feats.device,
                                    dtype=support_feats.dtype)
        for k in range(self.n_way):
            sup_mask = (support_labels == k)
            n_sup_k = sup_mask.sum().float()
            num = support_feats[sup_mask].sum(0) + (probs[:, k:k+1] * query_feats).sum(0)
            denom = n_sup_k + probs[:, k].sum()
            refined_proto[k] = num / (denom + 1e-8)

        return self.metric(query_feats, refined_proto)

    def _compute_tos_loss(self, x, z_support, z_query):
        """
        Temporal Order Sensitivity auxiliary loss.
        L_tos = mean cosine(feat(x), feat(flip_t(x)))  — MINIMIZE to push features apart.

        Rationale: SSv2 TAMT features have js_mean=0.02 under temporal flip
        (our own TempCTPR diagnostic) — nearly order-invariant. TOS forces
        TAA to encode temporal structure: original vs reversed video must
        produce dissimilar features.

        Reuses z_support, z_query already computed for the main CE loss.
        """
        x_rev = torch.flip(x, dims=[3])   # flip temporal axis
        z_s_rev, z_q_rev, _, _ = self.parse_feature(x_rev, is_feature=False)

        z_orig = torch.cat([
            z_support.contiguous().view(-1, z_support.shape[-1]),
            z_query.contiguous().view(-1, z_query.shape[-1])], dim=0)
        z_rev = torch.cat([
            z_s_rev.contiguous().view(-1, z_s_rev.shape[-1]),
            z_q_rev.contiguous().view(-1, z_q_rev.shape[-1])], dim=0)

        cos = F.cosine_similarity(z_orig, z_rev, dim=-1)    # [B]
        return cos.mean(), z_orig, z_rev

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)

        if getattr(self.params, 'use_tcmt', False):
            warmup = getattr(self, 'current_epoch', 0) < getattr(self.params, 'tcmt_warmup_epochs', 0)
            scores = self.set_forward_tcmt(
                x, tau=getattr(self.params, 'tcmt_tau', 1.0),
                detach_probs=getattr(self.params, 'tcmt_detach_probs', True),
                warmup=warmup)
            ce_loss = self.loss_fn(scores, y_query)
            total_loss = ce_loss
        elif getattr(self.params, 'use_tos', False):
            # Forward once, get features + CE loss, then compute TOS on flipped pass
            z_s, z_q, _, _ = self.parse_feature(x, is_feature=False)
            z_proto = z_s.contiguous().view(self.n_way, self.n_support, -1).mean(1)
            z_query_flat = z_q.contiguous().view(self.n_way * self.n_query, -1)
            scores = self.metric(z_query_flat, z_proto)
            ce_loss = self.loss_fn(scores, y_query)

            tos_warmup = getattr(self, 'current_epoch', 0) < getattr(
                self.params, 'tos_warmup_epochs', 0)
            if tos_warmup:
                total_loss = ce_loss
            else:
                tos_cos, _, _ = self._compute_tos_loss(x, z_s, z_q)
                lam = getattr(self.params, 'tos_lambda', 0.1)
                total_loss = ce_loss + lam * tos_cos
                # Track per-epoch cos stats
                ep = getattr(self, 'current_epoch', 0)
                last_ep = getattr(self, '_tos_last_logged_ep', -1)
                if ep != last_ep:
                    print(f'[TOS] ep {ep} first batch: CE={ce_loss.item():.3f} '
                          f'cos(orig,rev)={tos_cos.item():.4f} λ={lam}')
                    self._tos_last_logged_ep = ep
        else:
            scores = self.set_forward(x)
            total_loss = self.loss_fn(scores, y_query)

        # >>> DDAT auxiliary task (additive, orthogonal to head_variant/TCMT/TOS)
        # Extracts backbone tokens (PRE-BDC) for x and flip(x), trains classifier
        # to distinguish forward vs reversed. Forces backbone (or TAA inside) to
        # encode temporal direction. Uses backbone tokens BEFORE the symmetric BDC
        # head — that's the only place where direction info still exists.
        if getattr(self.params, 'use_ddat', False):
            ddat_warmup = getattr(self, 'current_epoch', 0) < getattr(
                self.params, 'ddat_warmup_epochs', 2)
            if not ddat_warmup:
                lam = getattr(self.params, 'ddat_lambda', 0.3)
                ddat_loss, ddat_acc = self._compute_ddat_loss(x)
                total_loss = total_loss + lam * ddat_loss
                ep = getattr(self, 'current_epoch', 0)
                last_ep = getattr(self, '_ddat_last_logged_ep', -1)
                if ep != last_ep:
                    print(f'[DDAT] ep {ep} first batch: L_dir={ddat_loss.item():.3f} '
                          f'dir_acc={ddat_acc:.3f} λ={lam}')
                    self._ddat_last_logged_ep = ep
        # <<< end DDAT

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), total_loss, scores

    def _backbone_tokens(self, x):
        """Run backbone forward and return [B, T, C] spatially-pooled tokens.
        x: [way, sample, C, T, H, W] (6D) or [B, C, T, H, W] (5D).
        T_tokens = 8 (VideoMAE patch16x2 default).
        """
        if x.dim() == 6:
            B = x.shape[0] * x.shape[1]
            x = x.contiguous().view(B, *x.size()[2:])
        x = x.cuda()
        # backbone forward — returns (x_cls, y_tokens) where y is [B, N, C]
        _, y = self.feature.forward(x)
        # y: [B, N=T*H*W_tok, C]; for 112-res with VideoMAE-S: N = 8*7*7 = 392
        B, N, C = y.shape
        T = 8
        spatial = N // T
        # reshape to [B, T, spatial, C], spatial-mean, output [B, T, C]
        y = y.view(B, T, spatial, C).mean(dim=2)
        return y    # [B, T, C]

    def _compute_ddat_loss(self, x):
        """Direction-discrimination loss: predict 0=forward, 1=reversed
        from spatially-pooled backbone tokens (PRE-BDC)."""
        x_flip = torch.flip(x, dims=[3])  # 6D: flip temporal axis (dim 3 = T)
        tok_orig = self._backbone_tokens(x)        # [B, T, C]
        tok_flip = self._backbone_tokens(x_flip)   # [B, T, C]
        # Flatten [B, T, C] → [B, T·C]
        feat_orig = tok_orig.flatten(1)
        feat_flip = tok_flip.flatten(1)
        # Stack and predict
        feats = torch.cat([feat_orig, feat_flip], dim=0)        # [2B, T·C]
        labels = torch.cat([
            torch.zeros(feat_orig.size(0), dtype=torch.long, device=feats.device),
            torch.ones(feat_flip.size(0), dtype=torch.long, device=feats.device),
        ])
        logits = self.direction_classifier(feats)               # [2B, 2]
        loss = self.loss_fn(logits, labels)
        with torch.no_grad():
            acc = (logits.argmax(-1) == labels).float().mean().item()
        return loss, acc

    def forward_meta_val_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.val_n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.val_n_way), self.n_query)
        scores = self.forward_meta_val(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        # print('x',x.shape) #x torch.Size([80, 32896])
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        # print('x',x.shape) #x torch.Size([80, 5, 32896])

        if self.n_support > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            score = (x * y).sum(2)
        # print('score',score.shape) #score torch.Size([80, 5])
        return score
    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score
