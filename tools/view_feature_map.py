'''
Author: Shuailin Chen
Created Date: 2021-11-18
Last Modified: 2021-11-18
	content: adapted from ghw
'''
import os.path as osp
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import PIL
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import tqdm
import argparse
from mmcv import Config

from openselfsup.models import builder


def view_feature_map(backbone_cfg, pth_name, img_path, video_dir='tmp', fram_size=(480, 480)):
    ''' View feature map of pretrained backbone in MP4 format'''
    net = builder.build_backbone(backbone_cfg)
    net.init_weights(pretrained=f'work_dirs/{pth_name}.pth')
    net.eval()
    # print(net)

    print(f'reading image {img_path}')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    train_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = train_transform(img).unsqueeze(0)
    with torch.no_grad():
        c2, c3, c4, c5 = net(img)

    c2 = c2.numpy()
    c3 = c3.numpy()
    c4 = c4.numpy()
    c5 = c5.numpy()
    # c2 = c2[0,0].numpy()
    # c3 = c3[0,0].numpy()
    # c4 = c4[0,0].numpy()
    # c5 = c5[0,0].numpy()

    # plt.matshow(c2)
    # plt.savefig('./vis_test/c2_{}.jpg'.format(exp_name))
    # plt.matshow(c3)
    # plt.savefig('./vis_test/c3_{}.jpg'.format(exp_name))
    # plt.matshow(c4)
    # plt.savefig('./vis_test/c4_{}.jpg'.format(exp_name))
    # plt.matshow(c5)
    # plt.savefig('./vis_test/c5_{}.jpg'.format(exp_name))

    target = c5
    videoPath = osp.join(video_dir, f'{pth_name.replace(r"/", "_")}_c5.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(videoPath, fourcc=fourcc, fps=30, frameSize=fram_size)

    for i in tqdm.trange(512):
        plt.matshow(target[0,i])
        buffer_ = BytesIO()
        plt.savefig(buffer_, format = 'jpg')
        plt.close()
        buffer_.seek(0)
        dataPIL = PIL.Image.open(buffer_)
        data = np.asarray(dataPIL, dtype=np.uint8)
        buffer_.close()
        videoWriter.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', help='train config file path')
    args = parser.parse_args()

    backbone_cfg = dict(
                    type='ResNetV1c',
                    depth=50,
                    in_channels=3,
                    out_indices=[1,2,3,4],
                    norm_cfg=dict(type='BN'),
                    dilations=(1, 1, 2, 4),
                    strides=(1, 2, 1, 1))

    # cfg = Config.fromfile(args.config)
    # backbone_cfg = cfg.model.backbone
    pth_name = 'pbyolv3_r50-d8-pretrained_sn6_sar_pro_ul_fh_v2_ep1600_lars_lr02_bs256/20211108_102442/mmseg_epoch_1600'
    img_path = 'data/SN6_sup/SAR-PRO_rotated/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804111224_20190804111453_tile_8691.tif'
    video_dir = 'tmp'
    view_feature_map(backbone_cfg=backbone_cfg,
                    pth_name=pth_name,
                    img_path = img_path,
                    video_dir=video_dir)