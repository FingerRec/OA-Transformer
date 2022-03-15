"""
visualize both image + object + text
"""
import numpy as np
import cv2
from csv import reader
import os
import random
import matplotlib.pyplot as plt
import torch
import pdb
import textwrap
import pandas as pd
import json

full_csv = "MSRVTT/annotation/MSR_VTT.json"
data_source = "MSRVTT/videos/all"
feat_source = "MSRVTT/region_features_full"
output = "MSRVTT/region_visualization"


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    return frame_idxs


def read_frames_cv2(video_path, num_frames, sample='uniform', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    cap.release()
    return frames, success_idxs

# step1: open video

# step2: video seq

# step3: video feature


def tri_region_visualize(imgs, feat_paths, caption, outpath="visualization/1.png"):
    concat_imgs = None
    for i in [0, 3, 7]:
        frame1 = np.load(feat_paths[i], allow_pickle=True)
        boxes = frame1['bbox']
        features = frame1['x']  # 20 x 2048
        confident = frame1['info'].item()['objects_conf']
        # step 1: re-ranking the region with confidence
        object_ids = frame1['info'].item()['objects_id']
        condident_indices = np.argsort(confident)[::-1]
        boxes = boxes[condident_indices]
        features = features[condident_indices]
        object_ids = object_ids[condident_indices]
        confident = confident[condident_indices]
        new_object, unique_indices = np.unique(object_ids, return_index=True)
        # step 2: remove region with same object class
        boxes = boxes[unique_indices]
        features = features[unique_indices]
        object_ids = object_ids[unique_indices]
        # object_ids = object_ids[unique_indices]
        # confident = confident[unique_indices]
        # # print(boxes, features)
        # image_width = frame1['info'].item()['image_w']
        # image_height = frame1['info'].item()['image_h']
        # box_width = boxes[:, 2] - boxes[:, 0]
        # box_height = boxes[:, 3] - boxes[:, 1]
        # scaled_width = box_width / image_width
        # scaled_height = box_height / image_height
        # scaled_x = boxes[:, 0] / image_width
        # scaled_y = boxes[:, 1] / image_height
        # scaled_width = scaled_width[..., np.newaxis]
        # scaled_height = scaled_height[..., np.newaxis]
        # scaled_x = scaled_x[..., np.newaxis]
        # scaled_y = scaled_y[..., np.newaxis]
        # spatial_features = np.concatenate(
        #     (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), axis=1)
        # feat = torch.cat([torch.from_numpy(features), torch.from_numpy(spatial_features)], dim=1)
        classes = ['__background__']
        with open('utils/objects_vocab.txt', 'r') as f:
            for object in f.readlines():
                classes.append(object.split(',')[0].lower().strip())
        # print(features.shape)
        # plot top 5 objects
        img = imgs[i]
        if len(boxes) < 5:
            return False
        # print(img.shape)
        # print(img)
        colormap = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (155, 100, 100), (100, 155, 100)]
        for j in range(5):
            # print(boxes[j])
            cv2.putText(img, '%s: %s' % (classes[object_ids[j] + 1], confident[j]), (int(boxes[j][0]), int(boxes[j][1] + 15)),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        colormap[j],
                        1)
            cv2.rectangle(img, (int(boxes[j][0]), int(boxes[j][1])), (int(boxes[j][2]), int(boxes[j][3])),
                          colormap[j],
                          1)
        if concat_imgs is None:
            concat_imgs = img
        else:
            concat_imgs = np.concatenate((concat_imgs, img), axis=1)
    caption_img = np.ones((50, imgs[0].shape[1] * 3, 3)) * 255
    cv2.putText(caption_img, caption, (10, 10), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        colormap[j],
                        1)
    concat_imgs = np.concatenate((concat_imgs, caption_img), axis=0)
    cv2.imwrite(outpath, concat_imgs)
    return outpath


if __name__ == '__main__':

    f = open(full_csv)
    data = json.load(f)
    count = 0

    for row in data['annotations']:
        count += 1
        if count % 10 != 0:
            continue
        # if row['id'] % 200 != 1:
        #     continue
        if count > 2000:
            break
        video_path = os.path.join(data_source, row['image_id'] + '.mp4')
        imgs, success_idxs = read_frames_cv2(video_path, 8, 'uniform')
        feat_paths = [feat_source + '/' + row['image_id'] + '/' + str(success_idxs[k]) + '.npz' for k in range(8)]
        outpath = 'visualization/msrvtt_3f/{}_{}.jpg'.format(count, row['image_id'])
        tri_region_visualize(imgs, feat_paths, row['caption'], outpath)

