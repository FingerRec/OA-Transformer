import math
import numpy as np
import cv2


def patch_all_masks_from_bbox(bboxs, patch_rows=14):
    # generate patch masks from all bboxs
    # notice here bbox region is [1:3][0:2]
    patch_masks = np.zeros((patch_rows, patch_rows))
    bboxs[:, :4] = bboxs[:, :4] * patch_rows
    for index, bbox in enumerate(bboxs):
        if bbox[4] > 7 and bbox[5] > 7:
            bbox[0] += 1 / 3 * (bbox[2] - bbox[0])
            bbox[1] += 1 / 3 * (bbox[3] - bbox[1])
            bbox[2] -= 1 / 3 * (bbox[2] - bbox[0])
            bbox[3] -= 1 / 3 * (bbox[3] - bbox[1])
        patch_masks[int(bbox[1]):math.ceil(bbox[3]), int(bbox[0]):math.ceil(bbox[2])] = 1
    return patch_masks


def image_mask_from_bbox(bboxs, img_shape):
    # print(img_shape)
    print(bboxs)
    w, h = img_shape[1:]
    mask = np.zeros((w, h))
    for index, bbox in enumerate(bboxs):
        print(int(bbox[0].item()), int(bbox[2].item()), int(bbox[1].item()), int(bbox[3].item()))
        # # print(bbox)
        if bbox[4] > 0.5 and bbox[5] > 0.5:
            bbox[0] += 1 / 3 * (bbox[2] - bbox[0])
            bbox[1] += 1 / 3 * (bbox[3] - bbox[1])
            bbox[2] -= 1 / 3 * (bbox[2] - bbox[0])
            bbox[3] -= 1 / 3 * (bbox[3] - bbox[1])
        # print(bbox[0])
        bbox[0] = bbox[0] * w
        bbox[1] = bbox[1] * h
        bbox[2] = bbox[2] * w
        bbox[3] = bbox[3] * h
        mask[int(bbox[0].item()): int(bbox[2].item()), int(bbox[1].item()):int(bbox[3].item())] = 1
    print(mask)
    return mask


def visualize_mask(video, bboxs, out_path):
    """
    visualize three samples frames and show the masked videos
    Args:
        video:
        bboxs:
        out_path:

    Returns:

    """
    num_frames = len(video)
    out_imgs = None
    for index in range(num_frames):
        img = video[index] * 255.
        bbox_10 = bboxs[index]
        masks = image_mask_from_bbox(bbox_10, img.shape)
        mask_img = img *  masks
        if out_imgs is None:
            out_imgs = np.concatenate((img, mask_img), axis=2)
        else:
            out_imgs = np.concatenate((out_imgs, mask_img), axis=2)
    # print(out_imgs)
    # print(out_imgs.shape)
    out_imgs = np.moveaxis(out_imgs, 0, 2)
    # print(out_imgs)
    print(out_imgs.shape)
    cv2.imwrite('{}.png'.format(out_path), out_imgs)

# def visualize_mask(video, bboxs, out_path):
#     """
#     visualize three samples frames and show the masked videos
#     Args:
#         video:
#         bboxs:
#         out_path:
#
#     Returns:
#
#     """
#     num_frames = len(video)
#     out_imgs = None
#     for index in range(num_frames):
#         img = video[index] * 255.
#         img = img.permute(1, 2, 0)
#         print(img.shape)
#         img = cv2.resize(np.float32(img), (14, 14))
#         bbox_10 = bboxs[index]
#         # masks = image_mask_from_bbox(bbox_10, img.shape)
#         masks = patch_all_masks_from_bbox(bbox_10)
#         # print(masks)
#         mask_img = img * np.expand_dims(masks, axis=2)
#         if out_imgs is None:
#             out_imgs = np.concatenate((img, mask_img), axis=1)
#         else:
#             out_imgs = np.concatenate((out_imgs, mask_img), axis=1)
#     # print(out_imgs)
#     # print(out_imgs.shape)
#     # out_imgs = np.moveaxis(out_imgs, 0, 2)
#     # print(out_imgs)
#     print(out_imgs.shape)
#     cv2.imwrite('{}.png'.format(out_path), out_imgs)