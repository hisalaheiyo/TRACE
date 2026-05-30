"""
Stage-3 variant: whole-class supervised pretraining on the target base split.

TAMT goes K400 backbone -> episodic meta-tuning directly. Image CD-FSL work
(Tian et al. ECCV'20; Guo et al. ECCV'20) shows that whole-class supervised
pretraining yields embeddings that transfer BETTER cross-domain than episodic
meta-learning (which overfits the 5-way task structure).

This script trains the SAME architecture as meta_train.py (MetaDeepBDC:
backbone + TAA + GTMT head) but with a whole-class linear classifier over ALL
target base classes, using standard cross-entropy — NOT episodic.

FAIRNESS: uses the exact same target base split (videos + labels) as TAMT's
episodic meta-training. Only the supervision scheme differs. The whole-class
linear head is a TRAINING-ONLY device and is discarded at test time (test uses
the metric on novel-class prototypes, identical to TAMT).

Output: checkpoints_wcp/<dataset>_wcp.tar  with {'state': model.state_dict()}.
  - Variant B: test.py --model_path this  (whole-class only, no episodic)
  - Variant C: meta_train.py --init_ckpt this  (whole-class as episodic init)

Usage:
  python whole_class_pretrain.py --dataset SSv2Full --data_path filelist/SSv2Full \
      --pretrain_path model/112112vit-s-140epoch.pt \
      --out checkpoints_wcp/SSv2Full_wcp.tar --epoch 20
"""
import os, json, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.datamgr import SetDataManager
from methods.meta_deepbdc import MetaDeepBDC
from methods.protonet import ProtoNet
from utils import model_dict, load_model, set_gpu, set_seed


def extract_feat(model, x_flat):
    """Frozen backbone + TAA + avgpool on a flat video batch
    [B, C, T, H, W] -> [B, 384]. good_embed uses class_head='non' so
    parse_feature returns (z_support, z_query) (2 values, avgpool features)."""
    saved = (model.n_way, model.n_support, model.n_query)
    B = x_flat.shape[0]
    model.n_way, model.n_support, model.n_query = B, 1, 0
    try:
        out = model.parse_feature(x_flat.unsqueeze(1), is_feature=False)
        z_s = out[0]                       # 2-value (good_embed/protonet)
        feat = z_s.reshape(B, -1)
    finally:
        model.n_way, model.n_support, model.n_query = saved
    return feat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_path', required=True)
    p.add_argument('--pretrain_path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--image_size', default=112, type=int)
    p.add_argument('--reduce_dim', default=256, type=int)
    p.add_argument('--method', default='good_embed')   # avgpool head, class_head='non'
    p.add_argument('--model', default='VideoMAES')
    p.add_argument('--epoch', default=20, type=int)
    p.add_argument('--lr', default=5e-4, type=float)
    p.add_argument('--n_episode', default=600, type=int)
    p.add_argument('--n_query', default=15, type=int)
    p.add_argument('--seed', default=1, type=int)
    p.add_argument('--head_variant', default='tamt')
    args = p.parse_args()
    set_gpu(args)
    set_seed(args.seed)

    # --- Determine the base class label set & build a contiguous remap ---
    with open(os.path.join(args.data_path, 'base.json')) as f:
        base_meta = json.load(f)
    uniq = sorted(set(base_meta['image_labels']))
    label_remap = {orig: i for i, orig in enumerate(uniq)}
    num_classes = len(uniq)
    print(f'[WCP] {args.dataset}: {num_classes} base classes, '
          f'{len(base_meta["image_names"])} videos')

    # --- Video loader (episodic loader reused; we use the REAL labels) ---
    dm = SetDataManager(args.data_path, args.image_size, n_query=args.n_query,
                        n_episode=args.n_episode, json_read=True,
                        n_way=5, n_support=1)
    loader = dm.get_data_loader('base.json', aug=True)

    # --- Model: SAME frozen-backbone+TAA as TAMT/protonet, but avgpool head
    #     (good_embed). class_head='non' (set by method) -> avgpool features. ---
    model = ProtoNet(args, model_dict[args.model], n_way=5, n_support=1).cuda()
    model.train()
    model = load_model(model, args.pretrain_path)             # K400 backbone
    # avgpool feature dim (VideoMAE-S = 384); infer lazily from one forward.
    with torch.no_grad():
        _probe = next(iter(loader))[0].cuda().reshape(-1, *next(iter(loader))[0].shape[2:])[:2]
        feat_dim = extract_feat(model, _probe).shape[-1]
    head = nn.Linear(feat_dim, num_classes).cuda()            # whole-class head
    print(f'[GE] avgpool feat_dim={feat_dim}, {num_classes} base classes')

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)
    lossfn = nn.CrossEntropyLoss()

    print(f'[WCP] whole-class pretraining: {args.epoch} epochs')
    for epoch in range(args.epoch):
        t0 = time.time()
        total, correct, n = 0.0, 0, 0
        for x, target in loader:
            # x: [n_way, n_sample, C,T,H,W]; target: [n_way, n_sample] (real labels)
            x = x.cuda().reshape(-1, *x.shape[2:])            # [B, C,T,H,W]
            y = target.reshape(-1)                            # [B] real labels
            y = torch.tensor([label_remap[int(v)] for v in y], device=x.device)

            feat = extract_feat(model, x)                     # [B, 384]
            logits = head(feat)                               # [B, num_classes]
            loss = lossfn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            n += y.numel()
        print(f'[WCP] epoch {epoch}: loss={total/max(len(loader),1):.4f} '
              f'acc={100*correct/max(n,1):.2f}% ({(time.time()-t0)/60:.1f} min)')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Save ONLY the model state (TAA+GTMT). Whole-class head is discarded.
    torch.save({'state': model.state_dict(), 'epoch': args.epoch,
                'num_base_classes': num_classes}, args.out)
    print(f'[WCP] saved whole-class-pretrained checkpoint to {args.out}')
    print(f'[WCP] (linear head NOT saved — discarded as per CD-FSAR protocol)')


if __name__ == '__main__':
    main()
