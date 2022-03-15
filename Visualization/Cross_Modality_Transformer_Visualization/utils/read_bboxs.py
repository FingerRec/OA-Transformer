import numpy as np
import pickle
import torch


def read_bbox_from_pickle(object_path, top_k=5, v=1):
    frame1 = np.load(object_path, allow_pickle=True)
    boxes = frame1['bbox']
    # rank features and boxes according to confidence
    confident = frame1['info'].item()['objects_conf']
    condident_indices = np.argsort(confident)[::-1]
    boxes = boxes[condident_indices]
    object_ids = frame1['info'].item()['objects_id']
    if v == 2:
        new_object, unique_indices = np.unique(object_ids, return_index=True)
        # step 2: remove region with same object class
        boxes = boxes[unique_indices]
    # padding with same elements if not enough
    if boxes.shape[0] < top_k:
        res = top_k - boxes.shape[0]
        boxes = np.pad(boxes, (0, res), 'edge')
    boxes = boxes[:top_k, :]
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
    return torch.from_numpy(spatial_features)
