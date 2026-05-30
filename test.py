import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
from data.datamgr import SetDataManager

from methods.protonet import ProtoNet
from methods.good_embed import GoodEmbed
from methods.meta_deepbdc import MetaDeepBDC
from methods.stl_deepbdc import STLDeepBDC

from utils import *
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=112, type=int, choices=[112, 224])
parser.add_argument('--dataset', default='mini_imagenet', choices=['Rareact2','Rareact','d2iving48', 'k400','ucf101','hmdb51','tiered_imagenet', 'SSv2Full','SSv2Small','diving48'])
parser.add_argument('--data_path', type=str)
parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18', 'VideoMAE','VideoMAES','VideoMAEB','VideoMAES2','VideoMAENormal'])
parser.add_argument('--method', default='stl_deepbdc', choices=['meta_deepbdc', 'stl_deepbdc', 'stl','protonet', 'good_embed'])

parser.add_argument('--test_n_way', default=5, type=int, help='number of classes used for testing (validation)')
parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support')
parser.add_argument('--n_query', default=12, type=int, help='number of unlabeled data in each class during meta validation')
# parser.add_argument('--n_query', default=5, type=int, help='number of unlabeled data in each class')

parser.add_argument('--test_n_episode', default=2000, type=int, help='number of episodes in test')
parser.add_argument('--model_path', default='', help='meta-trained or pre-trained model .tar file path')
parser.add_argument('--test_task_nums', default=5, type=int, help='test numbers')
parser.add_argument('--gpu', default='0', help='gpu id')

parser.add_argument('--penalty_C', default=0.1, type=float, help='logistic regression penalty parameter')
parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimensions of BDC dimensionality reduction layer')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')

# TSRA (Task-Specific Residual Adapter) arguments
parser.add_argument('--use_tsra', action='store_true', default=False)
parser.add_argument('--tsra_iter', default=40, type=int)
parser.add_argument('--tsra_lr', default=0.5, type=float)
parser.add_argument('--tsra_form', default='lora', choices=['full', 'lora'])
parser.add_argument('--tsra_rank', default=16, type=int)
parser.add_argument('--tsra_distance', default='cos', choices=['cos', 'l2'])
parser.add_argument('--tsra_lam_entropy', default=0.0, type=float,
                    help='TENT-style query entropy weight (0=off, typical=1.0)')
parser.add_argument('--tsra_lam_wd', default=0.0, type=float)
parser.add_argument('--tsra_early_stop', default=0, type=int,
                    help='early stop iter (0=use max_iter)')

# CTPR (Confidence-Weighted Transductive Prototype Refinement)
parser.add_argument('--use_ctpr', action='store_true', default=False,
                    help='enable CTPR test-time prototype refinement')
parser.add_argument('--ctpr_threshold', default=0.5, type=float,
                    help='confidence threshold for HC pseudo-labeling (diag best: 0.5)')
parser.add_argument('--ctpr_iter', default=1, type=int)
parser.add_argument('--seed', default=0, type=int, help='random seed (for fair comparison)')

# BADC / DSPR
parser.add_argument('--use_badc', action='store_true', default=False)
parser.add_argument('--use_dspr', action='store_true', default=False, help='CTPR + BADC')
parser.add_argument('--base_stats', default='', type=str, help='path to cached base stats .pt')
parser.add_argument('--badc_k', default=3, type=int, help='number of nearest base classes')
parser.add_argument('--badc_nvirt', default=100, type=int, help='virtual samples per class')
parser.add_argument('--badc_alpha', default=0.21, type=float, help='variance shift (Free Lunch default)')

# Head variant (must match training-time setting; default 'tamt' = unchanged baseline)
parser.add_argument('--head_variant', default='tamt',
                    choices=['tamt', 'fodm', 'dar', 'causal', 'diff_stream', 'oatp'])

# Enhanced CTPR variants
parser.add_argument('--ctpr_mode', default='hard',
                    choices=['hard', 'iterative', 'soft', 'balanced', 'gctpr', 'tempctpr',
                             'sta_plain', 'sta_hard', 'sta_iter3',
                             'rpa_plain', 'rpa_hard', 'rpa_iter3', 'combo',
                             'trx', 'trx_ctpr', 'tcr', 'tcr_ctpr_ensemble',
                             'maf', 'maf_ens',
                             'trace', 'trace_ens'])
# MAF (Mutual-Agreement Filtering) — uses --ctpr_threshold, --ctpr_iter,
# and TCR hyperparams above. 'maf_ens' adds light TCR ensemble (--maf_ens_w).
parser.add_argument('--maf_ens_w', default=0.3, type=float,
                    help='TCR mix weight for maf_ens mode (0=pure MAF)')
# TRACE: blend factor for CTPR-init TCR. 1.0 = pure CTPR init (default),
# 0.0 = pure support init (= TCR-only).
parser.add_argument('--ctpr_init_w', default=1.0, type=float,
                    help='blend weight of CTPR proto in TCR init (1.0=pure CTPR, 0.0=pure support)')
parser.add_argument('--trx_temperature', default=0.1, type=float)
parser.add_argument('--trx_no_card2', action='store_true', default=False)
# TCR
parser.add_argument('--tcr_iter', default=100, type=int)
parser.add_argument('--tcr_lr', default=1e-3, type=float)
parser.add_argument('--tcr_alpha', default=0.1, type=float, help='conditional entropy weight')
parser.add_argument('--tcr_beta', default=0.1, type=float, help='marginal entropy weight')
parser.add_argument('--tcr_temp', default=15.0, type=float)
parser.add_argument('--tcr_metric', default='cos', choices=['cos', 'dot'])
# E: ensemble-weight sweep for tcr_ctpr_ensemble. Default 0.5 = previously validated.
parser.add_argument('--tcr_ctpr_w', default=0.5, type=float,
                    help='weight on TCR in tcr_ctpr_ensemble ensemble (1-w on CTPR)')

# Combo-1: NSAE channel weighting + episode standardization (test-time)
parser.add_argument('--use_nsae', default='', type=str,
                    help='path to cached NSAE channel weights .pt (empty = off)')
parser.add_argument('--use_standardize', action='store_true', default=False,
                    help='enable per-episode feature standardization')
parser.add_argument('--ctpr_topm', default=0, type=int, help='top-M per class (0=use threshold)')

# GCTPR
parser.add_argument('--gctpr_k', default=5, type=int, help='k for k-NN graph')
parser.add_argument('--gctpr_lp_alpha', default=0.5, type=float)
parser.add_argument('--gctpr_lp_iter', default=5, type=int)
parser.add_argument('--gctpr_no_agree', action='store_true', default=False,
                    help='do NOT require prop_label == softmax argmax')
parser.add_argument('--gctpr_query_init', default='softmax', choices=['uniform', 'softmax'],
                    help='Y_0 prior for queries: uniform or softmax (BD-CSPN-style)')

# TempCTPR
parser.add_argument('--tempctpr_js_thr', default=0.0, type=float)
parser.add_argument('--tempctpr_combine', default='and',
                    choices=['and', 'product', 'js_only'])

# Support-TTA (STA): reduce 1-shot support sampling noise by K-view averaging
parser.add_argument('--sta_K', default=4, type=int,
                    help='number of augmented support views (K=1 = no TTA)')

# RPA (Retrieval-based Prototype Augmentation)
parser.add_argument('--rpa_cache', default='', type=str,
                    help='path to cached base raw-features .pt (for RPA)')
parser.add_argument('--rpa_K', default=5, type=int,
                    help='top-K neighbors per support to retrieve')
parser.add_argument('--rpa_alpha', default=0.3, type=float,
                    help='blend weight for retrieved neighbor mean (mode=blend only)')
parser.add_argument('--rpa_mode', default='blend', choices=['blend', 'expand'],
                    help='blend = weighted avg; expand = add neighbors as new support')
parser.add_argument('--rpa_metric', default='cosine', choices=['cosine', 'l2'])

params = parser.parse_args()
num_gpu = set_gpu(params)

# Set seed for fair comparison (same episodes across baseline and CTPR runs)
if params.seed > 0:
    import random
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    print(f"[SEED] Set to {params.seed} for fair baseline/CTPR comparison")

json_file_read = False
if params.dataset == 'hmdb51':
        novel_file ='novel.json'
        json_file_read = True
elif params.dataset == 'ucf101':
        novel_file ='novel.json'
        json_file_read = True
elif params.dataset == 'SSv2Full':
        novel_file ='novel.json'
        json_file_read = True
elif params.dataset == 'SSv2Small':
        novel_file ='novel.json'
        json_file_read = True
elif params.dataset == 'diving48':
        novel_file ='novel.json'
        json_file_read = True
elif params.dataset == 'd2iving48':
        novel_file ='novel.json'
        json_file_read = True
elif params.dataset == 'Rareact':
        novel_file ='novel.json'
        json_file_read = True
elif params.dataset == 'Rareact2':
        novel_file ='novel.json'
        json_file_read = True
else:
    novel_file = 'test'


novel_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
novel_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.test_n_episode, json_read=json_file_read,  **novel_few_shot_params)
novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)

if params.method == 'protonet':
    model = ProtoNet(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'good_embed':
    model = GoodEmbed(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'meta_deepbdc':
    model = MetaDeepBDC(params, model_dict[params.model], **novel_few_shot_params)
elif params.method == 'stl_deepbdc':
    model = STLDeepBDC(params, model_dict[params.model], **novel_few_shot_params)

# model save path
model = model.cuda()
model.eval()

print(params.model_path)
model_file = os.path.join(params.model_path)
model = load_model(model, model_file)

# Initialize TSRA module if enabled (384-dim matching TAMT's z_all output)
tsra_module = None
if params.use_tsra and params.method == 'meta_deepbdc':
    from methods.tsra import TSRA
    tsra_module = TSRA(feat_dim=384, form=params.tsra_form, rank=params.tsra_rank).cuda()
    n_params = sum(p.numel() for p in tsra_module.parameters())
    print(f"[TSRA] Enabled: form={params.tsra_form}, rank={params.tsra_rank}, "
          f"iter={params.tsra_iter}, lr={params.tsra_lr}, distance={params.tsra_distance}")
    print(f"[TSRA] Params: {n_params:,}")
elif params.use_ctpr and params.method == 'meta_deepbdc':
    print(f"[CTPR] Enabled: threshold={params.ctpr_threshold}, iter={params.ctpr_iter}")
elif params.use_badc:
    base_stats = torch.load(params.base_stats, map_location='cuda', weights_only=False)
    print(f"[BADC] Enabled: k={params.badc_k}, n_virt={params.badc_nvirt}, alpha={params.badc_alpha}")
    print(f"[BADC] Loaded base stats: {base_stats['means'].shape[0]} classes")
elif params.use_dspr:
    base_stats = torch.load(params.base_stats, map_location='cuda', weights_only=False)
    print(f"[DSPR] CTPR+BADC: threshold={params.ctpr_threshold}, k={params.badc_k}, n_virt={params.badc_nvirt}, alpha={params.badc_alpha}")
    print(f"[DSPR] Loaded base stats: {base_stats['means'].shape[0]} classes")
else:
    print("[MODE] Baseline (no test-time adaptation)")

# RPA: load cached base features if mode starts with 'rpa_'
rpa_base = None
if params.ctpr_mode.startswith('rpa_'):
    assert params.rpa_cache, '--rpa_cache required for rpa_* modes'
    rpa_base = torch.load(params.rpa_cache, map_location='cuda', weights_only=False)
    print(f"[RPA] Loaded base features: {rpa_base['features'].shape} "
          f"from {rpa_base['class_ids'].shape[0]} classes")

# Combo-1: load cached NSAE channel weights if provided
nsae_w = None
if params.use_nsae:
    nsae_w = torch.load(params.use_nsae, map_location='cpu', weights_only=False)
    if isinstance(nsae_w, dict):
        nsae_w = nsae_w['weights']
    print(f"[NSAE] Loaded channel weights: shape={tuple(nsae_w.shape)}, "
          f"mean={nsae_w.mean():.3f}, min={nsae_w.min():.3f}, max={nsae_w.max():.3f}")
if params.use_standardize:
    print("[STD] Episode feature standardization enabled")

print(params)
iter_num = params.test_n_episode
acc_all_task = []
for _ in range(params.test_task_nums):
    acc_all = []
    test_start_time = time.time()
    tqdm_gen = tqdm.tqdm(novel_loader)
    for _, (x, _) in enumerate(tqdm_gen):
        model.n_query = params.n_query
        if params.use_dspr and params.method == 'meta_deepbdc':
            with torch.no_grad():
                scores = model.set_forward_with_dspr(
                    x, base_stats=base_stats,
                    conf_threshold=params.ctpr_threshold,
                    k_neighbors=params.badc_k,
                    n_virtual=params.badc_nvirt,
                    alpha=params.badc_alpha)
        elif params.use_badc and params.method == 'meta_deepbdc':
            with torch.no_grad():
                scores = model.set_forward_with_badc(
                    x, base_stats=base_stats,
                    k_neighbors=params.badc_k,
                    n_virtual=params.badc_nvirt,
                    alpha=params.badc_alpha)
        elif params.use_ctpr and params.method == 'meta_deepbdc':
            if params.ctpr_mode == 'iterative':
                scores = model.set_forward_with_ctpr_iterative(
                    x, conf_threshold=params.ctpr_threshold, n_iter=params.ctpr_iter)
            elif params.ctpr_mode == 'soft':
                scores = model.set_forward_with_ctpr_soft(x)
            elif params.ctpr_mode == 'balanced':
                top_m = params.ctpr_topm if params.ctpr_topm > 0 else None
                scores = model.set_forward_with_ctpr_balanced(
                    x, conf_threshold=params.ctpr_threshold, top_m_per_class=top_m)
            elif params.ctpr_mode == 'gctpr':
                scores = model.set_forward_with_gctpr(
                    x, conf_threshold=params.ctpr_threshold, n_iter=params.ctpr_iter,
                    k_neighbors=params.gctpr_k, lp_alpha=params.gctpr_lp_alpha,
                    lp_iter=params.gctpr_lp_iter,
                    require_agreement=not params.gctpr_no_agree,
                    query_init=params.gctpr_query_init)
            elif params.ctpr_mode == 'tempctpr':
                scores = model.set_forward_with_tempctpr(
                    x, conf_threshold=params.ctpr_threshold, n_iter=params.ctpr_iter,
                    js_threshold=params.tempctpr_js_thr,
                    combine=params.tempctpr_combine)
            elif params.ctpr_mode == 'sta_plain':
                scores = model.set_forward_with_ctpr_sta(
                    x, K=params.sta_K, use_ctpr=False)
            elif params.ctpr_mode == 'sta_hard':
                scores = model.set_forward_with_ctpr_sta(
                    x, K=params.sta_K, conf_threshold=params.ctpr_threshold, n_iter=1)
            elif params.ctpr_mode == 'sta_iter3':
                scores = model.set_forward_with_ctpr_sta(
                    x, K=params.sta_K, conf_threshold=params.ctpr_threshold, n_iter=3)
            elif params.ctpr_mode == 'rpa_plain':
                scores = model.set_forward_with_rpa(
                    x, base_features=rpa_base, K=params.rpa_K, alpha=params.rpa_alpha,
                    mode=params.rpa_mode, use_ctpr=False, metric=params.rpa_metric)
            elif params.ctpr_mode == 'rpa_hard':
                scores = model.set_forward_with_rpa(
                    x, base_features=rpa_base, K=params.rpa_K, alpha=params.rpa_alpha,
                    mode=params.rpa_mode, use_ctpr=True, n_iter=1,
                    conf_threshold=params.ctpr_threshold, metric=params.rpa_metric)
            elif params.ctpr_mode == 'rpa_iter3':
                scores = model.set_forward_with_rpa(
                    x, base_features=rpa_base, K=params.rpa_K, alpha=params.rpa_alpha,
                    mode=params.rpa_mode, use_ctpr=True, n_iter=3,
                    conf_threshold=params.ctpr_threshold, metric=params.rpa_metric)
            elif params.ctpr_mode == 'combo':
                scores = model.set_forward_with_ctpr_combo(
                    x, conf_threshold=params.ctpr_threshold, n_iter=params.ctpr_iter,
                    nsae_weights=nsae_w, use_standardize=params.use_standardize)
            elif params.ctpr_mode == 'trx':
                scores = model.set_forward_with_trx(
                    x, temperature=params.trx_temperature,
                    use_card2=not params.trx_no_card2, combine_ctpr=False)
            elif params.ctpr_mode == 'trx_ctpr':
                scores = model.set_forward_with_trx(
                    x, temperature=params.trx_temperature,
                    use_card2=not params.trx_no_card2, combine_ctpr=True,
                    conf_threshold=params.ctpr_threshold, n_iter=params.ctpr_iter)
            elif params.ctpr_mode == 'tcr':
                scores = model.set_forward_with_tcr(
                    x, n_iter=params.tcr_iter, lr=params.tcr_lr,
                    alpha=params.tcr_alpha, beta=params.tcr_beta,
                    temp=params.tcr_temp, metric=params.tcr_metric, combine_ctpr=False)
            elif params.ctpr_mode == 'tcr_ctpr_ensemble':
                scores = model.set_forward_with_tcr(
                    x, n_iter=params.tcr_iter, lr=params.tcr_lr,
                    alpha=params.tcr_alpha, beta=params.tcr_beta,
                    temp=params.tcr_temp, metric=params.tcr_metric, combine_ctpr=True,
                    conf_threshold=params.ctpr_threshold, ctpr_n_iter=params.ctpr_iter,
                    tcr_ctpr_w=params.tcr_ctpr_w)
            elif params.ctpr_mode == 'maf':
                scores = model.set_forward_with_maf(
                    x, conf_threshold=params.ctpr_threshold,
                    n_iter=params.ctpr_iter,
                    tcr_n_iter=params.tcr_iter, tcr_lr=params.tcr_lr,
                    tcr_alpha=params.tcr_alpha, tcr_beta=params.tcr_beta,
                    tcr_temp=params.tcr_temp, tcr_metric=params.tcr_metric,
                    ensemble_w=0.0)
            elif params.ctpr_mode == 'maf_ens':
                scores = model.set_forward_with_maf(
                    x, conf_threshold=params.ctpr_threshold,
                    n_iter=params.ctpr_iter,
                    tcr_n_iter=params.tcr_iter, tcr_lr=params.tcr_lr,
                    tcr_alpha=params.tcr_alpha, tcr_beta=params.tcr_beta,
                    tcr_temp=params.tcr_temp, tcr_metric=params.tcr_metric,
                    ensemble_w=params.maf_ens_w)
            elif params.ctpr_mode == 'trace':
                scores = model.set_forward_with_trace(
                    x, conf_threshold=params.ctpr_threshold,
                    ctpr_n_iter=params.ctpr_iter,
                    tcr_n_iter=params.tcr_iter, tcr_lr=params.tcr_lr,
                    tcr_alpha=params.tcr_alpha, tcr_beta=params.tcr_beta,
                    tcr_temp=params.tcr_temp, tcr_metric=params.tcr_metric,
                    combine_ctpr=False, ctpr_init_w=params.ctpr_init_w)
            elif params.ctpr_mode == 'trace_ens':
                scores = model.set_forward_with_trace(
                    x, conf_threshold=params.ctpr_threshold,
                    ctpr_n_iter=params.ctpr_iter,
                    tcr_n_iter=params.tcr_iter, tcr_lr=params.tcr_lr,
                    tcr_alpha=params.tcr_alpha, tcr_beta=params.tcr_beta,
                    tcr_temp=params.tcr_temp, tcr_metric=params.tcr_metric,
                    combine_ctpr=True, tcr_ctpr_w=params.tcr_ctpr_w,
                    ctpr_init_w=params.ctpr_init_w)
            else:  # 'hard' — original
                with torch.no_grad():
                    scores = model.set_forward_with_ctpr(
                        x, conf_threshold=params.ctpr_threshold,
                        n_iter=params.ctpr_iter)
        elif tsra_module is not None:
            scores = model.set_forward_with_tsra(
                x, tsra_module,
                max_iter=params.tsra_iter, lr=params.tsra_lr,
                distance=params.tsra_distance,
                lam_entropy=params.tsra_lam_entropy,
                lam_wd=params.tsra_lam_wd,
                early_stop_iter=params.tsra_early_stop if params.tsra_early_stop > 0 else None)
        else:
            with torch.no_grad():
                scores = model.set_forward(x, False)
        if params.method in ['meta_deepbdc', 'protonet']:
            pred = scores.data.cpu().numpy().argmax(axis=1)
        else:
            pred = scores
        y = np.repeat(range(params.test_n_way), params.n_query)
        acc = np.mean(pred == y) * 100

        acc_all.append(acc)
        tqdm_gen.set_description(f'avg.acc:{(np.mean(acc_all)):.2f} (curr:{acc:.2f})')

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%% (Time uses %.2f minutes)'
        % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num), (time.time() - test_start_time) / 60))
    acc_all_task.append(acc_all)

acc_all_task_mean = np.mean(acc_all_task)
print('%d test mean acc = %4.2f%%' % (params.test_task_nums, acc_all_task_mean))

