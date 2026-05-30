import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

from data.datamgr import SetDataManager

from methods.protonet import ProtoNet
from methods.meta_deepbdc import MetaDeepBDC
from utils import *


def train(params, base_loader, val_loader, model, stop_epoch):

    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    # SGD and Adam have similar effects
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=5e-4)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=0.0000001)#余弦衰减

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    for epoch in range(0, stop_epoch):
        start = time.time()
        model.train()
        model.current_epoch = epoch
        trainObj, top1 = model.train_loop(epoch, base_loader, optimizer)

        model.eval()
        valObj, acc = model.test_loop(val_loader)
        if acc > trlog['max_acc']:
            print("best model! save...")
            trlog['max_acc'] = acc
            trlog['max_acc_epoch'] = epoch
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        trlog['train_loss'].append(trainObj)
        trlog['train_acc'].append(top1)
        trlog['val_loss'].append(valObj)
        trlog['val_acc'].append(acc)
        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        lr_scheduler.step()

        print("This epoch use %.2f minutes" % ((time.time() - start) / 60))
        print("train loss is {:.2f}, train acc is {:.2f}".format(trainObj, top1))
        print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
        print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', default=224, type=int, choices=[112, 224], help='input image size, 84 for miniImagenet and tieredImagenet, 224 for cub')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate of the backbone')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 80], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=100, type=int, help='Stopping epoch')
    parser.add_argument('--gpu', default='0', help='gpu id')

    parser.add_argument('--dataset', default='mini_imagenet', choices=['Rareact2','d2iving48','Rareact', 'k400','ucf101','hmdb51','SSv2Full','SSv2Small','tiered_imagenet', 'diving48'])
    parser.add_argument('--data_path', type=str, help='dataset path')

    parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18', 'VideoMAENormal','VideoMAES','VideoMAES2','VideoMAEB','VideoMAE'])
    parser.add_argument('--tunning_mode', default='normal', choices=['normal', 'PSRP', 'SSF','ss'])
    parser.add_argument('--method', default='meta_deepbdc', choices=['meta_deepbdc', 'stl','protonet'])

    parser.add_argument('--train_n_episode', default=600, type=int, help='number of episodes in meta train')
    parser.add_argument('--val_n_episode', default=300, type=int, help='number of episodes in meta val')
    parser.add_argument('--train_n_way', default=5, type=int, help='number of classes used for meta train')
    parser.add_argument('--val_n_way', default=5, type=int, help='number of classes used for meta val')
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=12, type=int, help='number of unlabeled data in each class')
    # parser.add_argument('--n_query', default=5, type=int, help='number of unlabeled data in each class')
    parser.add_argument('--distributed', action='store_true', default=True)

    parser.add_argument('--extra_dir', default='', help='record additional information')

    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in pretrain')
    parser.add_argument('--pretrain_path', default='', help='pre-trained model .tar file path')
    parser.add_argument('--init_ckpt', default='', type=str,
                        help='warmed checkpoint (SSL stage-2 output) to init full model from; '
                             'if set, overrides pretrain_path (which only loads K400 backbone)')
    parser.add_argument('--save_freq', default=10, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--reduce_dim', default=640, type=int, help='the output dimension of BDC dimensionality reduction layer')

    # Head variant — escapes TAMT's symmetric-covariance head limitation.
    # Diagnostic: js_mean=0.02 on SSv2 → MPNCOV/BDC are mathematically symmetric →
    #   temporal-flip information is destroyed by second-order statistics.
    # Three escape designs (mutually exclusive, default 'tamt' = original unchanged):
    #   tamt   : original TAMT head (baseline, bit-exact identical to before)
    #   fodm   : First-Order Difference Moments — adds mean(δ_t) directional vector
    #   dar    : Dual sym + GRU directional branch with learnable α fusion
    #   causal : pure causal-Transformer head (replaces symmetric branch entirely)
    parser.add_argument('--head_variant', default='tamt',
                        choices=['tamt', 'fodm', 'dar', 'causal', 'diff_stream', 'oatp'],
                        help='head architecture variant for escaping symmetric-covariance limitation')

    # DDAT (Direction-Discrimination Auxiliary Task): force backbone tokens to
    # encode forward/reversed direction via a small classifier on backbone tokens.
    # Loss: L = CE_episode + λ_ddat · L_direction
    # Operates on PRE-BDC tokens (post-BDC features are symmetric → useless).
    parser.add_argument('--use_ddat', action='store_true', default=False,
                        help='enable Direction-Discrimination Auxiliary Task')
    parser.add_argument('--ddat_lambda', default=0.3, type=float,
                        help='weight of direction loss (typical 0.1-0.5)')
    parser.add_argument('--ddat_warmup_epochs', default=2, type=int,
                        help='epochs at start without DDAT (CE-only) for stability')

    # TCMT (Transductive-Consistent Meta-Training): soft-CTPR in the training loop
    parser.add_argument('--use_tcmt', action='store_true', default=False,
                        help='enable TCMT: soft-CTPR refined prototypes during training')
    parser.add_argument('--tcmt_tau', default=1.0, type=float,
                        help='softmax temperature for soft-CTPR (higher=smoother)')
    parser.add_argument('--tcmt_warmup_epochs', default=2, type=int,
                        help='epochs at start using plain forward (no refine) to stabilize')
    parser.add_argument('--tcmt_detach_probs', action='store_true', default=True,
                        help='detach pseudo-soft-labels from graph (EM-style, more stable)')

    # TOS (Temporal Order Sensitivity): force features to differ under temporal flip
    parser.add_argument('--use_tos', action='store_true', default=False,
                        help='enable TOS auxiliary loss: L = CE + λ · cos(feat(x), feat(flip_t(x)))')
    parser.add_argument('--tos_lambda', default=0.1, type=float,
                        help='TOS loss weight (smaller = less aggressive)')
    parser.add_argument('--tos_warmup_epochs', default=2, type=int,
                        help='epochs of CE-only warmup before enabling TOS')

    params = parser.parse_args()
    num_gpu = set_gpu(params)                                  

    set_seed(params.seed)
    
    

    json_file_read = False
    if params.dataset == 'Rareact':
        base_file = 'base.json'
        val_file = 'val.json'
        json_file_read = True
        params.num_classes = 64
    elif params.dataset == 'Rareact2':
        base_file = 'base.json'
        val_file = 'val.json'
        json_file_read = True
        params.num_classes = 64
    elif params.dataset == 'diving48':
        base_file = 'base.json'
        val_file = 'val.json'
        json_file_read = True
        params.num_classes = 48
    elif params.dataset == 'd2iving48':
        base_file = 'base.json'
        val_file = 'val.json'
        json_file_read = True
        params.num_classes = 48
    elif params.dataset == 'k400':
        base_file = 'VideoMAEv2base.json'
        val_file = 'VideoMAEv2val.json'
        json_file_read = True
        params.num_classes = 400
    elif params.dataset == 'hmdb51':
        base_file = 'base.json'
        val_file =  'val.json'
        json_file_read = True
        params.num_classes = 51
    elif params.dataset == 'ucf101':
        base_file = 'base.json'
        val_file =  'val.json'
        json_file_read = True
        params.num_classes = 101
    elif params.dataset == 'SSv2Full':
        base_file = 'base.json'
        val_file =  'val.json'
        json_file_read = True
    else:
        ValueError('dataset error')

    train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    base_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.train_n_episode, json_read=json_file_read, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
    val_datamgr = SetDataManager(params.data_path, params.image_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

    if params.method == 'protonet':
        model = ProtoNet(params, model_dict[params.model], **train_few_shot_params)
    elif params.method == 'meta_deepbdc':
        model = MetaDeepBDC(params, model_dict[params.model], **train_few_shot_params)

    model = model.cuda()

    # model save path
    params.checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    params.checkpoint_dir += '_2TAA' # TAA 10-11
    if params.use_tcmt:
        params.checkpoint_dir += '_tcmt_tau%.1f_wu%d' % (params.tcmt_tau, params.tcmt_warmup_epochs)
    if params.use_tos:
        params.checkpoint_dir += '_tos_lam%.2f_wu%d' % (params.tos_lambda, params.tos_warmup_epochs)
    if params.head_variant != 'tamt':
        params.checkpoint_dir += '_head_%s' % params.head_variant
    if params.init_ckpt:
        params.checkpoint_dir += '_sslwarm'
    if params.use_ddat:
        params.checkpoint_dir += '_ddat_lam%.2f_wu%d' % (params.ddat_lambda, params.ddat_warmup_epochs)
    params.checkpoint_dir += params.extra_dir
    print(params.checkpoint_dir)

    model = model.cuda()
    
    dir1 = ''
    if params.init_ckpt:
        # Stage-2 warmed checkpoint: load full model state (backbone + TAA + heads)
        print(f'[INIT] loading warmed checkpoint: {params.init_ckpt}')
        model = load_model(model, params.init_ckpt)
    else:
        print(params.pretrain_path)
        modelfile = os.path.join(params.pretrain_path)
        model = load_model(model, modelfile)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params)
    model = train(params, base_loader, val_loader, model, params.epoch)
