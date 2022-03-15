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

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    print("have load")

from torch.multiprocessing import Value
load_video_count = Value('i', 0)
load_frames_count = Value('i', 0)


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


def sample_frames(num_frames, vlen):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    return frame_idxs


def read_frames_cv2(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(video_path, vlen)
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
    cap.release()
    return frames, success_idxs

# Extract features.
# image_dir structure: [dir_name]/[video_name]/[*.png]
# out_dir structure: [dir_name]/[video_name]/[*.npz]


def extract_video_feat_tmp(gpu_index, worker_index, video_files, cfg, args, model):
    print('*'*100)
    videoNum = len(video_files)
    print("model init finished on gpu: {} process: {}".format(gpu_index, worker_index))
    corNum = 0
    pre = args.output_dir
    for i in range(videoNum):
        cls_dir = pre + video_files[i].split('/')[-3]
        if not os.path.exists(cls_dir):
            os.mkdir(cls_dir)
        video_name = args.video_dir + '/' + video_files[i].split('/')[-3] + '/' + video_files[i].split('/')[-2] + '.mp4'
        # vidcap = cv2.VideoCapture(video_name)
        # vlen = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(video_name, vlen)
        # frame_indices = list(np.linspace(0, vlen - 1, num=args.sampling_frames, dtype=np.int))
        video_dir = cls_dir + '/' + video_files[i].split('/')[-2]
        try:
            frames, _ = read_frames_cv2(video_name, args.sampling_frames)
            # frames, _ = read_frames_av_fast(video_name, args.sampling_frames)
        except AssertionError:
            print("load video {} failed".format(video_name))
            continue
        flag = 1
        for index in range(len(frames)):
            im = frames[index]
            # print(im.shape)
            out_dir = video_dir + '/{}.npz'.format(index)
            # if already exists frame, just continue
            if os.path.exists(out_dir):
                with load_frames_count.get_lock():
                    load_frames_count.value += 1
                if index == len(frames) - 1:
                    corNum += 1
                    flag = 0
                    with load_video_count.get_lock():
                        load_video_count.value += 1
                    if corNum % 1 == 0:
                        print("gpu {} process {} : {}/{} video already exist".format(gpu_index, worker_index, corNum,
                                                                                     videoNum))
                continue
            if im is None:
                print(video_name, "is illegal!")
                continue
            dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
            # extract roi features
            attr_scores = None
            with torch.set_grad_enabled(False):
                if cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                else:
                    boxes, scores, features_pooled = model([dataset_dict])
            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            # print(boxes)
            if not attr_scores is None:
                attr_scores = [attr_score.cpu() for attr_score in attr_scores]
            try:
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
            except FileExistsError:
                print("file already exists")
                continue
            except PermissionError:
                print("permission error")
                continue
            try:
                generate_npz(
                    args, cfg, video_name, out_dir, im, dataset_dict,
                    boxes, scores, features_pooled, attr_scores)
            except PermissionError:
                continue
            # print("process 1 image")
        with load_video_count.get_lock():
            load_video_count.value += 1
        if flag == 1:
            corNum += 1
        if corNum % 1 == 0:
            print("gpu {} process {} : {}/{} videos extract features".format(gpu_index, worker_index, corNum, videoNum))
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument('--num-cpus', default=1, type=int,
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

    parser.add_argument('--sampling_frames', default=1, type=int, help="the uniform sampling frames per video")

    parser.add_argument('--split', default="val", type=str, help="train or val")

    parser.add_argument('--output_dir', default="", type=str, help="the output dir")

    parser.add_argument('--dataset_dir', default="WebVid", type=str, help="the output dir")

    parser.add_argument('--workers_per_gpu', default=1, type=int, help="process per gpu")
    args = parser.parse_args()

    cfg = setup(args)

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # num_gpus = len(args.gpu_id.split(','))

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    args.video_dir = "{}/{}/".format(args.dataset_dir, args.split)
    args.output_dir = "{}/{}_frame_object/{}/".format(args.dataset_dir, args.sampling_frames, args.split)

    # model = model_init(cfg, args)
    # model.share_memory()
    # videoList, outList = load_video_path(args)
    # alex: load  videoList and outList from dict
    with open("webvid_8_frame_object_loss_3.txt", "rb") as fp:   # Unpickling
        videoList = pickle.load(fp)
    # with open("webvid_8_frame_object_loss_2.txt", "rb") as fp:   # Unpickling
    #     videoList = pickle.load(fp)
    # with open("inputs_train.txt", "rb") as fp:   # Unpickling
    #     videoList = pickle.load(fp)
    processes = []
    num_gpus = len(args.gpu_id.split(','))
    print("use {} gpus".format(num_gpus))
    num_processes = num_gpus * args.workers_per_gpu
    videoList = videoList[::-1]
    video_lists = [videoList[i::num_processes] for i in range(num_processes)]

    gpus = args.gpu_id.split(',')
    # num_processes can be as much as possible
    torch.set_num_threads(128)
    model = None
    model = model_init(cfg, args)
    model.share_memory()
    for rank in range(num_processes):
        if rank % args.workers_per_gpu == 0:
            print('*' * 200)
            print("load model on GPU:{}".format(str(gpus[rank // args.workers_per_gpu])))
            print('*' * 200)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[rank // args.workers_per_gpu])
            model = model.cuda()
        p = mp.Process(target=extract_video_feat_tmp,
                       args=(
                       gpus[rank // args.workers_per_gpu], rank % args.workers_per_gpu, video_lists[rank], cfg, args,
                       model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()