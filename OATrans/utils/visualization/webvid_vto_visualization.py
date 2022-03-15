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

full_csv = "WebVid2M_videos/metadata/results_subset_train.csv"
data_source = "WebVid2M_videos/train_videos"
feat_source = "WebVid2M_frames_region_features/train"
output = "WebVid2M_visualization/train"


def feature_visualize(img1, feat_path):
    frame1 = np.load(feat_path, allow_pickle=True)
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
    confident = confident[unique_indices]

    # print(boxes, features)
    image_width = frame1['info'].item()['image_w']
    image_height = frame1['info'].item()['image_h']

    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    scaled_width = box_width / image_width
    scaled_height = box_height / image_height
    scaled_x = boxes[:, 0] / image_width
    scaled_y = boxes[:, 1] / image_height
    scaled_width = scaled_width[..., np.newaxis]
    scaled_height = scaled_height[..., np.newaxis]
    scaled_x = scaled_x[..., np.newaxis]
    scaled_y = scaled_y[..., np.newaxis]
    spatial_features = np.concatenate(
        (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height), axis=1)
    # print(spatial_features)
    feat = torch.cat([torch.from_numpy(features), torch.from_numpy(spatial_features)], dim=1)
    classes = ['__background__']
    with open('../objects_vocab.txt', 'r') as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())
    # print(features.shape)
    im = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(im)
    new_boxes = boxes
    for i in range(len(new_boxes)):
        bbox = new_boxes[i]
        if i < 10:
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=2, alpha=0.5)
            )
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '%s: %s' % (classes[object_ids[i] + 1], confident[i]),
                           bbox=dict(facecolor='blue', alpha=0.5),
                           fontsize=10, color='white')
    outpath = "test_roi.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


with open(full_csv, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    count = 0
    for row in csv_reader:
        count += 1
        if count == 1:
            continue
        # if count > 3:
        #     break
        # cv2.destroyAllWindows()
        if len(row[3]) < 3:
            continue
        video_path = os.path.join(data_source, row[3], row[0] + '.mp4')
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        feat_path = os.path.join(feat_source, row[3], row[0], '1.npz')
        feat_img = cv2.imread(feature_visualize(img, feat_path))
        print(feat_img.shape)
        caption_img = np.ones([feat_img.shape[0]//4, feat_img.shape[1], 3]) * 255

        wrapped_text = textwrap.wrap(row[1], width=35)
        x, y = 10, 40
        font_size = 1
        font_thickness = 2
        font = cv2.FONT_HERSHEY_TRIPLEX

        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

            gap = textsize[1] + 10

            y = 25 + i * gap
            x = int((caption_img.shape[1] - textsize[0]) / 2) + 20

            cv2.putText(caption_img, line, (x, y), font,
                        font_size,
                        (122, 21, 91),
                        font_thickness,
                        lineType=cv2.LINE_AA)

        # cv2.putText(caption_img, row[1], (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        #                     1)
        concat = np.concatenate([feat_img[50:-50, :, :], caption_img], axis=0)
        # cv2.imshow(concat)
        out_file = os.path.join(output, row[3] + row[0] + '.png')
        print("hello world")
        cv2.imwrite(out_file, concat)
        # if cv2.waitKey(33) == 27:
        #     continue
