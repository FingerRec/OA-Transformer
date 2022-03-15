# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pylint: disable=no-member
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys
import torch
# import tqdm
import cv2
import numpy as np
sys.path.append('detectron2')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob, alex_save_roi_features
from utils.progress_bar import ProgressBar
from models import add_config
from models.bua.box_regression import BUABoxes
import time
import torch.multiprocessing as mp
from itertools import product
import pickle
import pandas as pd
from PIL import Image

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    print("have load")

from torch.multiprocessing import Value
load_image_count = Value('i', 0)

def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd


def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes,
            'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def generate_npz(*args):
    alex_save_roi_features(*args)


def model_init(cfg, args):
    print('*'*100)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    # model.set_device("cuda:"+index)
    model.eval()
    return model

# Extract features.
# image_dir structure: [dir_name]/[video_name]/[*.png]
# out_dir structure: [dir_name]/[video_name]/[*.npz]


def extract_image_feat_tmp(gpu_index, worker_index, image_files, cfg, args, model):
    print('*'*100)
    imageNum = len(image_files)
    print("model init finished on gpu: {} process: {}".format(gpu_index, worker_index))
    # load the overall dataset
    corNum = 0
    pre = "CC3M/1_frame_object/train"
    # pre = "CC3M/1_frame_object/train"
    # print(image_files)
    for i in range(imageNum):
        image_dir = pre + '/' + image_files[i].split('/')[-2]
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        out_dir = image_dir + '/1'
        # print(out_dir)
        if os.path.exists(out_dir + '.npz'):
            corNum += 1
            with load_image_count.get_lock():
                load_image_count.value += 1
            print("gpu {} process {}: {}/{} already exist".format(gpu_index, worker_index, corNum, imageNum))
            # print("{} already exist".format(outList[i]))
            continue
        # print(image_files[i])
        im_file = "CC3M/training/" + image_files[i].split('/')[-2]
        # im_file = "CC3M/training/" + image_files[i]
        im = Image.open(im_file).convert("RGB")
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        # im = cv2.imread(im_file)

        dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
        # extract roi features
        attr_scores = None
        try:
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
        except RuntimeError:
            print("{} is too small".format(im_file))
            continue
        # print(boxes)
        if not attr_scores is None:
            attr_scores = [attr_score.cpu() for attr_score in attr_scores]
        try:
            generate_npz(
                args, cfg, image_files[i], out_dir, im, dataset_dict,
                boxes, scores, features_pooled, attr_scores)
        except PermissionError:
            print("{} can't be write due to permission".format(out_dir))
            continue
        corNum += 1
        print("gpu {} process {} : {}/{} images extract features".format(gpu_index, worker_index, corNum, imageNum))
        end_time = time.time()
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=32, type=int,
                        help='number of cpus to use for ray, 0 means no limit')

    parser.add_argument('--gpus', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)

    parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")

    parser.add_argument('--extract-mode', default='roi_feats', type=str,
                        help="'roi_feats', 'bboxes' and 'bbox_feats' indicates \
                        'extract roi features directly', 'extract bboxes only' and \
                        'extract roi features with pre-computed bboxes' respectively")

    parser.add_argument('--min-max-boxes', default='min_max_default', type=str,
                        help='the number of min-max boxes of extractor')
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = setup(args)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # num_gpus = len(args.gpu_id.split(','))

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH
    # model = model_init(cfg, args)
    # model.share_memory()
    # videoList, outList = load_video_path(args)
    # with open("cc3m_val_full.txt", "rb") as fp:   # Unpickling
    #     videoList = pickle.load(fp)
    # metadata = pd.read_csv("cc3m_training_success_full.tsv", sep='\t')
    # print("{} images will be processed".format(len(metadata)))
    # videoList = [metadata.iloc[item][1] for item in range(len(metadata))]
    with open("cc3m_object_loss_3.txt", "rb") as fp:   # Unpickling
        videoList = pickle.load(fp)
    processes = []
    num_gpus = len(args.gpu_id.split(','))
    workers_per_gpu = 2
    num_processes = num_gpus * workers_per_gpu
    videoList = videoList[::-1]
    print("{} videos need to be processed".format(len(videoList)))
    video_lists = [videoList[i::num_processes] for i in range(num_processes)]

    gpus = args.gpu_id.split(',')
    # num_processes can be as much as possible
    model = None
    for rank in range(num_processes):
        if rank % workers_per_gpu == 0:
            print('*'*200)
            print("load model on GPU:{}".format(str(rank//workers_per_gpu)))
            print('*'*200)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[rank//workers_per_gpu])
            model = model_init(cfg, args)
            model = model.cuda()
        p = mp.Process(target=extract_image_feat_tmp, args=(gpus[rank//workers_per_gpu], rank, video_lists[rank], cfg, args, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
