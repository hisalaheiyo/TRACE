"""
TUB-Clf: Classifier-side Transductive Upper Bound (for paper motivation).

Motivation: the existing oracle (diagnose_full.py D1) is PROTOTYPE-side — it
bounds metric/prototype methods (e.g., CTPR). For TCR, which optimizes a
*classifier* W on support + query, the natural upper bound is "what if we
TRAINED W with the true query labels?". Naively training on support + ALL
labeled query and evaluating on the same query is circular (test = train).

This script implements an unbiased classifier-side upper bound:

  Per episode:
    1. Extract frozen TAMT features (same as baseline / CTPR / TCR).
    2. Repeat S random 50/50 splits of the query set:
         half_A : "labeled" — used together with support to train an LR head
         half_B : held-out  — used for evaluation only
       Train multinomial Logistic Regression on (support ∪ half_A) with TRUE
       labels; report accuracy on half_B.
    3. Average over S splits → per-episode TUB-Clf accuracy.

  Also reports for direct comparison on the same episodes (same seed, same
  feature extractor):
    - acc_base:  prototype baseline (TAMT metric).
    - acc_proto_oracle: prototype-side oracle (D1) — bound on the SAME
       held-out half_B, using support ∪ half_A with TRUE labels and TAMT's
       metric (so it's directly comparable to acc_tub_clf with no train/test
       overlap).
    - acc_tcr_oracle_init: NEW — runs TCR (info-max optimizer) with W
       initialized from the ORACLE prototype (support ∪ half_A with TRUE
       labels), then evaluates on half_B. Bounds TRACE:
       motivates the design "improving TCR's initialization lifts TCR's
       final accuracy". TCR here uses the validated hyperparams from
       methods/tcr.py (alpha=0.1, beta=1.0, iter=100, lr=1e-3, temp=15, cos).

  This gives the cleanest paper picture:
    baseline ≤ acc_proto_oracle (bounds CTPR)
            ≤ acc_tub_clf       (bounds TCR, generally tighter)

Fairness / correctness:
  - Uses model.parse_feature exactly like test.py (frozen features).
  - LR uses sklearn (deterministic given seed). Features L2-normalized so
    cosine geometry (matching TCR's metric) is preserved.
  - Same episode RNG as test.py (--seed 42, 2000 episodes × 5 tasks
    available; default 600 episodes for speed).
  - Pure analysis. Does NOT touch any checkpoint / RESULTS.md / existing
    diagnose outputs.

Outputs:
  JSON file at logs/tub_clf_<dataset>_<shot>shot.json containing per-episode
  numbers and the means/stds at the end.

Usage:
  python tub_clf_oracle.py --dataset SSv2Full --data_path filelist/SSv2Full \\
      --model_path checkpoints/.../best_model.tar \\
      --n_shot 1 --n_episodes 600 --n_splits 20 --seed 42
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', category=ConvergenceWarning)

from data.datamgr import SetDataManager
from methods.meta_deepbdc import MetaDeepBDC
from methods.tcr import tcr_refine
from utils import model_dict, load_model, set_gpu


@torch.no_grad()
def extract_episode_features(model, x, n_way, n_support, n_query):
    """Frozen TAMT features. See diagnose_full.py:extract_episode_features."""
    z_s, z_q, _, _ = model.parse_feature(x, is_feature=False)
    support_feats = z_s.contiguous().view(n_way * n_support, -1)
    query_feats = z_q.contiguous().view(n_way * n_query, -1)
    return support_feats, query_feats


def proto_score(s_feats, s_labels, q_feats, n_way, metric):
    """Prototype scoring matching TAMT (dot for 1-shot, l2 for multi-shot)."""
    prots = torch.zeros(n_way, s_feats.shape[-1],
                        device=s_feats.device, dtype=s_feats.dtype)
    for k in range(n_way):
        m = (s_labels == k)
        if m.sum() > 0:
            prots[k] = s_feats[m].mean(0)
    q = q_feats.unsqueeze(1)
    p = prots.unsqueeze(0)
    if metric == 'l2':
        return -((q - p) ** 2).sum(-1)
    elif metric == 'dot':
        return (q * p).sum(-1)
    else:
        raise ValueError(metric)


def lr_eval(train_x, train_y, eval_x, eval_y, seed):
    """L2-normalized multinomial logistic regression, leave-one-half-out style.
    Returns accuracy in [0, 1]."""
    # L2 normalize so LR works on cosine geometry (matches TCR's metric).
    eps = 1e-12
    tx = train_x / (np.linalg.norm(train_x, axis=1, keepdims=True) + eps)
    ex = eval_x  / (np.linalg.norm(eval_x,  axis=1, keepdims=True) + eps)
    # Multinomial LR. C=10 is a mild regularization; deterministic with seed.
    # Note: 'lbfgs' + multi-class problem -> multinomial automatically in
    # sklearn >=1.5 (the explicit multi_class kwarg was removed/deprecated).
    clf = LogisticRegression(
        C=10.0, max_iter=1000, solver='lbfgs',
        random_state=seed, n_jobs=1,
    )
    clf.fit(tx, train_y)
    pred = clf.predict(ex)
    return float((pred == eval_y).mean())


@torch.no_grad()
def diagnose_episode(model, x, n_way, n_support, n_query, metric,
                      n_splits, ep_seed):
    """Compute baseline + prototype oracle + TUB-Clf on a single episode.

    For oracle/TUB-Clf, we average accuracy over n_splits random 50/50
    splits of the query set. Train side: (support ∪ half_A, true labels).
    Eval side: half_B (true labels).
    """
    s_feats, q_feats = extract_episode_features(
        model, x, n_way, n_support, n_query)
    device = s_feats.device

    # Labels (TAMT episode layout: queries are in row-major class order).
    s_labels = torch.arange(n_way, device=device).repeat_interleave(n_support)
    q_labels = torch.arange(n_way, device=device).repeat_interleave(n_query)

    # ----- baseline: prototype on support, eval on ALL queries -----
    sc_base = proto_score(s_feats, s_labels, q_feats, n_way, metric)
    acc_base_all = (sc_base.argmax(1) == q_labels).float().mean().item()

    # ----- splits (same RNG per episode for reproducibility) -----
    rng = np.random.RandomState(ep_seed)
    s_feats_np = s_feats.cpu().numpy()
    q_feats_np = q_feats.cpu().numpy()
    s_labels_np = s_labels.cpu().numpy()
    q_labels_np = q_labels.cpu().numpy()
    N_q = q_feats.shape[0]

    base_split, proto_split, tub_split, tcr_init_split = [], [], [], []
    for split_idx in range(n_splits):
        perm = rng.permutation(N_q)
        half = N_q // 2
        idx_A = perm[:half]
        idx_B = perm[half:]
        qa_f, qa_y = q_feats_np[idx_A], q_labels_np[idx_A]
        qb_f, qb_y = q_feats_np[idx_B], q_labels_np[idx_B]

        # ----- baseline restricted to half_B (for apples-to-apples) -----
        acc_b_half = float(
            (sc_base[idx_B].argmax(1).cpu().numpy() == qb_y).mean())
        base_split.append(acc_b_half)

        # ----- prototype oracle: support ∪ half_A with TRUE labels -----
        train_f_t = torch.from_numpy(
            np.concatenate([s_feats_np, qa_f], 0)).to(device)
        train_y_t = torch.from_numpy(
            np.concatenate([s_labels_np, qa_y], 0)).to(device)
        qb_f_t = torch.from_numpy(qb_f).to(device)
        sc_o = proto_score(train_f_t, train_y_t, qb_f_t, n_way, metric)
        proto_split.append(
            float((sc_o.argmax(1).cpu().numpy() == qb_y).mean()))

        # ----- TUB-Clf: LR on support ∪ half_A; eval on half_B -----
        tub_split.append(lr_eval(
            np.concatenate([s_feats_np, qa_f], 0),
            np.concatenate([s_labels_np, qa_y], 0),
            qb_f, qb_y, seed=ep_seed + split_idx))

        # ----- TUB-TCR-Init: TCR with oracle init, eval on half_B -----
        # Build oracle proto from support ∪ half_A; pass to TCR as W_init.
        # TCR optimizes using support (labels=true) + half_B (unlabeled queries).
        # Bound on TRACE: best possible "init-improved TCR".
        # Use validated TCR hyperparams (matching methods/tcr.py defaults from
        # the validated tcr_ctpr_ensemble / trace setup).
        P_oracle = torch.zeros(n_way, s_feats.shape[-1],
                                device=device, dtype=s_feats.dtype)
        all_train_f_t = train_f_t   # already on device
        all_train_y_t = train_y_t
        for k in range(n_way):
            m = (all_train_y_t == k)
            if m.sum() > 0:
                P_oracle[k] = all_train_f_t[m].mean(0)
        # TCR uses support (with TRUE labels) and half_B as the unlabeled query
        # set. half_A is NOT shown to TCR (it's only "consumed" by the oracle
        # init); half_B is the held-out test set.
        with torch.enable_grad():
            tcr_logits_b = tcr_refine(
                s_feats, s_labels, qb_f_t, n_way=n_way,
                n_iter=100, lr=1e-3, alpha=0.1, beta=1.0,
                temp=15.0, metric='cos', init_W=P_oracle,
            )
        tcr_init_split.append(
            float((tcr_logits_b.argmax(1).cpu().numpy() == qb_y).mean()))

    return {
        'acc_base_all':       acc_base_all,
        'acc_base_halfB':     float(np.mean(base_split)),
        'acc_proto_oracle':   float(np.mean(proto_split)),
        'acc_tub_clf':        float(np.mean(tub_split)),
        'acc_tcr_oracle_init': float(np.mean(tcr_init_split)),
    }


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
    p.add_argument('--n_query', default=12, type=int,
                    help='MUST match test.py default (12) so episodes are '
                         'identical to baseline/CTPR/TCR/MAF runs.')
    p.add_argument('--n_episodes', default=600, type=int)
    p.add_argument('--n_splits', default=20, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--gpu', default='0')
    p.add_argument('--out_dir', default='logs')
    args = p.parse_args()

    # ---- Seed & GPU (mirror test.py) ----
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_gpu(args)

    # ---- TAMT metric (matches test.py / meta_deepbdc.metric) ----
    metric = 'dot' if args.n_shot == 1 else 'l2'

    # ---- Data: same loader as test.py ----
    novel_json = 'novel.json'
    dm = SetDataManager(args.data_path, args.image_size,
                        n_query=args.n_query, n_episode=args.n_episodes,
                        json_read=True, n_way=args.n_way,
                        n_support=args.n_shot)
    loader = dm.get_data_loader(novel_json, aug=False)

    # ---- Model: same as test.py ----
    model = MetaDeepBDC(args, model_dict[args.model],
                        n_way=args.n_way, n_support=args.n_shot).cuda()
    model.eval()
    model = load_model(model, args.model_path)

    print(f'[TUB-Clf] {args.dataset} {args.n_shot}-shot, '
          f'{args.n_episodes} episodes × {args.n_splits} splits, metric={metric}')
    print(f'[TUB-Clf] ckpt = {args.model_path}')

    results = []
    t0 = time.time()
    tqdm_gen = tqdm.tqdm(loader, total=args.n_episodes)
    for ep_idx, (x, _) in enumerate(tqdm_gen):
        x = x.cuda()
        # mirror test.py: n_way, n_support implicit via model attributes
        model.n_query = args.n_query
        info = diagnose_episode(
            model, x,
            n_way=args.n_way, n_support=args.n_shot, n_query=args.n_query,
            metric=metric, n_splits=args.n_splits,
            ep_seed=args.seed + ep_idx,
        )
        results.append(info)
        if (ep_idx + 1) % 50 == 0:
            arr = lambda k: 100 * np.mean([r[k] for r in results])
            tqdm_gen.set_description(
                f'base={arr("acc_base_all"):.2f} '
                f'protoO={arr("acc_proto_oracle"):.2f} '
                f'TUBclf={arr("acc_tub_clf"):.2f}')

    # ---- Aggregate ----
    keys = ['acc_base_all', 'acc_base_halfB',
            'acc_proto_oracle', 'acc_tub_clf', 'acc_tcr_oracle_init']
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in results]) * 100
        summary[k] = {'mean': float(vals.mean()),
                      'std':  float(vals.std()),
                      'sem':  float(vals.std() / np.sqrt(len(vals)))}
    elapsed = time.time() - t0

    # ---- Output ----
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
        f'tub_clf_{args.dataset}_{args.n_shot}shot.json')
    with open(out_path, 'w') as f:
        json.dump({
            'dataset': args.dataset, 'n_shot': args.n_shot,
            'n_episodes': args.n_episodes, 'n_splits': args.n_splits,
            'seed': args.seed, 'metric': metric,
            'model_path': args.model_path,
            'elapsed_sec': elapsed,
            'summary': summary,
            'per_episode': results,
        }, f, indent=2)

    print('\n========== TUB-Clf Summary ==========')
    print(f'{args.dataset} {args.n_shot}-shot, {args.n_episodes} eps × {args.n_splits} splits')
    print(f'baseline (all queries):        {summary["acc_base_all"]["mean"]:.2f} ± {summary["acc_base_all"]["sem"]:.2f}')
    print(f'baseline (half_B only):        {summary["acc_base_halfB"]["mean"]:.2f} ± {summary["acc_base_halfB"]["sem"]:.2f}')
    print(f'proto oracle (TAMT metric):    {summary["acc_proto_oracle"]["mean"]:.2f} ± {summary["acc_proto_oracle"]["sem"]:.2f}')
    print(f'TUB-Clf (LR, cosine geom):     {summary["acc_tub_clf"]["mean"]:.2f} ± {summary["acc_tub_clf"]["sem"]:.2f}')
    print(f'TUB-TCR-Init (TCR w/ orcl init):{summary["acc_tcr_oracle_init"]["mean"]:.2f} ± {summary["acc_tcr_oracle_init"]["sem"]:.2f}')
    gap_proto = summary['acc_proto_oracle']['mean']    - summary['acc_base_halfB']['mean']
    gap_tub   = summary['acc_tub_clf']['mean']         - summary['acc_base_halfB']['mean']
    gap_tinit = summary['acc_tcr_oracle_init']['mean'] - summary['acc_base_halfB']['mean']
    print(f'\nproto-oracle gap (bounds CTPR):       +{gap_proto:.2f}')
    print(f'TUB-Clf gap     (bounds TCR):         +{gap_tub:.2f}')
    print(f'TUB-TCR-Init gap (bounds TRACE):     +{gap_tinit:.2f}')
    print(f'elapsed: {elapsed/60:.1f} min')
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
