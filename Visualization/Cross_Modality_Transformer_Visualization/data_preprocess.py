import cv2
import numpy as np
import random
from torchvision import transforms
import torch
from PIL import Image, ImageFile


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


def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None, numpy=False):
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
        # print(frame.shape)
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    if not numpy:
        frames = torch.stack(frames).float() / 255
        cap.release()
    return frames, success_idxs, vlen


def vision_preprocess(vid_src):
    video, _, _ = read_frames_cv2(vid_src, 1)
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            # transforms.RandomResizedCrop(size=(224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    video = transform(video)
    # expand one dim as batch
    video = video.unsqueeze(0)
    return video.cuda()


def mask_vision_preprocess(vid_src):
    video, _, _ = read_frames_cv2(vid_src, 8)
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            # transforms.RandomResizedCrop(size=(224, 224)),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    video = transform(video[:3])
    return video


def vision_img_preprocess(img_src):
    img = Image.open(img_src).convert("RGB")
    img = transforms.ToTensor()(img).unsqueeze(0)
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            # transforms.RandomResizedCrop(size=(224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    img = transform(img)
    # expand one dim as batch
    img = img.unsqueeze(0)
    return img.cuda()


def clip_img_preprocess(img_src, preprocess):
    img = Image.open(img_src).convert("RGB")
    img = preprocess(img).unsqueeze(0)
    # print(img.size())
    return img.cuda().half()