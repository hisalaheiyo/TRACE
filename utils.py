import numpy as np
import os
import glob
import argparse
import network.resnet as resnet
import network.VideoMAE as VideoMAE
import torch
import random
from weight_loaders import weight_loader_fn_dict

model_dict = dict(
    ResNet10=resnet.ResNet10,
    ResNet12=resnet.ResNet12,
    ResNet18=resnet.ResNet18,
    ResNet34=resnet.ResNet34,
    ResNet34s=resnet.ResNet34s,
    ResNet50=resnet.ResNet50,
    ResNet101=resnet.ResNet101,
    VideoMAEB=VideoMAE.vit_base_patch16_224,
    VideoMAES=VideoMAE.vit_small_patch16_112,
    VideoMAES2=VideoMAE.vit_small_patch16_224,
    VideoMAEGiant=VideoMAE.vit_giant_patch14_224
    )


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    print(best_file)
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def load_model(model, dir):
    # Match file type by basename (not hardcoded full path)
    basename = os.path.basename(dir)

    # load pretrained model of SL (K400 SL-trained backbone)
    if basename in ('112112vit-s-140epoch.pt', 'vit-s-120epoch.pt', '112112vit-s-120epoch.pt'):
        file_dict = torch.load(dir, map_location='cpu', weights_only=False)['module']
        model.feature.load_state_dict(file_dict, strict=False)
        return model

    # load pretrained model of SSL (VideoMAE SSL checkpoint)
    if basename.endswith('.pth') and 'wo' in basename.lower():
        file_dict = torch.load(dir, map_location='cpu', weights_only=False)['model']
        model.feature.load_state_dict(file_dict, strict=False)
        return model

    # load finetuned model (.tar checkpoint with {'state': state_dict})
    model_dict = model.state_dict()
    file_dict = torch.load(dir, map_location='cpu', weights_only=False)['state']
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False