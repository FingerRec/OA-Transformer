import random
import cv2
import av
import os
import numpy as np
import torch
import random
from PIL import Image, ImageFile
from abc import abstractmethod
import tarfile
from io import BytesIO
from torchvision import transforms
from torch.utils.data import Dataset, get_worker_info
import sys
import pickle
import decord
import matplotlib.pyplot as plt
import math
# import torchvision.transforms as transforms
# import nltk
# nltk.data.path.append("pretrained/nltk_data")
# from textaugment import EDA

def textaug_eda(caption):
    aug_caption = caption
    t = EDA()
    if len(caption) < 15:
        return aug_caption
    try:
        if random.random() < 0.5:
            if random.random() < 0.3:
                aug_caption = t.synonym_replacement(aug_caption)
            if random.random() < 0.3:
                aug_caption = t.random_swap(aug_caption)
            if random.random() < 0.3:
                aug_caption = t.random_insertion(aug_caption)
            aug_caption = t.random_deletion(aug_caption, p=random.random()*0.3)
    except AttributeError:
        print("{} / {} attribute error".format(caption, aug_caption))
    return str(aug_caption)

def textaug_advanced(caption, aug_model):
    return aug_model.augment(caption)


def mask_aug(sentence):
    words = sentence.split(' ')
    word_index = random.randint(0, len(words))
    words[word_index] = "[MASK]"
    new_cpation = ' '.join(words)
    new_sentence = ""
    # shuffle object localization
    # random drop some objects
    return new_sentence


def aug_object(sentence, thresh=0.3):
    p = random.random() * thresh
    words = sentence.split(' ')
    for i in range(len(words)):
        if random.random() < p:
            words[i] = "[MASK]"
    new_sentence = ' '.join(words)
    return new_sentence


def mask_object_tags(sentence, index):
    words = sentence.split(' ')
    if index > len(words):
        return sentence
    words[index] = "[MASK]"
    new_sentence = ' '.join(words)
    return new_sentence

def aug_object_del(sentence, thresh=0.4):
    p = random.random() * thresh
    words = sentence.split(' ')
    for i in range(len(words)):
        if random.random() < p:
            words[i] = ""
    new_sentence = ' '.join(words)
    return new_sentence

def shuffle_object(sentence):
    words = sentence.split(' ')
    # print(words)
    # print(len(words))
    arr = np.arange(len(words))
    np.random.shuffle(arr)
    # print(arr)
    new_sentence = ""
    for i in range(len(words)):
        new_sentence += ' ' + words[arr[i]]
    # print(new_sentence)
    return new_sentence

def add_pseudo_class(sentence, vocab="utils/objects_vocab_fine_grained.txt"):
    classes = []
    with open(vocab, 'r') as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())
    new_object = np.random.randint(1599, size=10)
    object_tags = ""
    for n in range(0, len(new_object)):
        object_tags += ' ' + (classes[new_object[n]])
    new_sentence = shuffle_object(sentence + object_tags)
    return new_sentence

class TextVideoDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2'
                 ):
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError("Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)
        #assert os.path.exists(video_fp)
        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'rand'
        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']

        try:
            imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample, fix_start=fix_start)
        except:
            if video_loading == 'strict':
                raise ValueError(f'Video loading failed for {video_fp}, video loading for this dataset is strict.')
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr}
        return data

class TextObjectVideoDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 text_params,
                 object_params,
                 video_params,
                 data_dir,
                 object_dir,
                 metadata_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 mask=False
                 ):
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        self.object_params = object_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        # == object dir ==
        self.object_dir = os.path.expandvars(object_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        print("read video with {}".format(reader))
        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self._load_metadata()
        self.mask = mask
        print("&"*100)
        print(self.transforms)
        print("&" * 100)
        if self.split == 'train':
            self.transforms = transforms.Compose(
                [
                transforms.Resize(size=(224, 224)),
                # transforms.RandomResizedCrop(size=(224, 224)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        # if self.dataset_name == 'WebVid':
        #     if self.text_params['object_tags']:
        #         with open('utils/data_preprocess/webvid/object_tags_full_{}.pickle'.format(split), 'rb') as handle:
        #             self.object_tags_dicts = pickle.load(handle)
        #     if self.object_params['input_objects']:
        #         with open('base/data_preprocess/webvid/subset_{}.pickle'.format(split), 'rb') as handle:
        #             self.object_dicts = pickle.load(handle)
        #     if self.object_params['pseudo_labels']:
        #         with open('base/data_preprocess/webvid/object_labels_subset_{}.pickle'.format(split), 'rb') as handle:
        #             self.pseudo_label_dicts = pickle.load(handle)
        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()
        # if self.object_params['text_aug']:
        #     from gensim.models import KeyedVectors
        #     model = KeyedVectors.load_word2vec_format('pretrained/GoogleNews-vectors-negative300.bin.gz', binary=True)
        #     from textaugment import Word2vec
        #     self.text_aug_model = Word2vec(model=model, runs=5, v=False, p=0.5)

        self.object_token_lens = np.loadtxt("utils/objects_vocab_token_len.txt")
        vocab = "utils/objects_vocab.txt"
        self.classes = ['__background__']
        with open(vocab, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError("Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_object_path(self, sample, index=0, rm_split=False):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def read_object_tags(self, object_path):
        # pdb.set_trace()
        return self.object_tags_dicts[object_path]

    def read_pseudo_label(self, object_path):
        # pdb.set_trace()
        # gt_labels = self.pseudo_label_dicts[object_path]
        gt_labels = read_object_labels_from_disk(object_path)
        pad_labels = torch.zeros(1601)
        for i in range(len(gt_labels)):
            pad_labels[int(gt_labels[i])] = 1
        return pad_labels

    def patch_mask_from_bbox(self, bboxs, patch_rows=14):
        # print(bboxs)
        patch_indexs = np.zeros(patch_rows*patch_rows)
        bboxs[:, :4] = bboxs[:, :4] * patch_rows
        for index, bbox in enumerate(bboxs):
            # if the object is too large, just keep center
            # if bbox[2] - bbox[0] > 7 and bbox[3] - bbox[1] > 7:
            #     bbox[0] += 1 / 3 * (bbox[2] - bbox[0])
            #     bbox[1] += 1 / 3 * (bbox[3] - bbox[1])
            #     bbox[2] -= 1 / 3 * (bbox[2] - bbox[0])
            #     bbox[3] -= 1 / 3 * (bbox[3] - bbox[1])
            for i in range(patch_rows):
                for j in range(patch_rows):
                    if i > bbox[0] and i < bbox[2] and j > bbox[1] and j < bbox[3]:
                        patch_indexs[i*patch_rows+j] = 1
        # print(np.sum(patch_indexs)/196)
        return patch_indexs

    def patch_all_masks_from_bbox(self, bboxs, patch_rows=14):
        # generate patch masks from all bboxs
        # notice here bbox region is [1:3][0:2]
        patch_masks = np.zeros((len(bboxs), patch_rows, patch_rows))
        bboxs[:, :4] = bboxs[:, :4] * patch_rows
        for index, bbox in enumerate(bboxs):
            bbox = bboxs[index]
            patch_masks[index, int(bbox[1]):math.ceil(bbox[3]), int(bbox[0]):math.ceil(bbox[2])] = 1
        return np.reshape(patch_masks, (len(bboxs), patch_rows * patch_rows))

    def image_mask_from_bbox(self, bboxs, img_shape):
        # print(bboxs)
        ## mask background information
        mask = torch.zeros((img_shape[0], img_shape[1]))
        bboxs[:, 0] = bboxs[:, 0] * img_shape[0]
        bboxs[:, 1] = bboxs[:, 1] * img_shape[1]
        bboxs[:, 2] = bboxs[:, 2] * img_shape[0]
        bboxs[:, 3] = bboxs[:, 3] * img_shape[1]
        for index, bbox in enumerate(bboxs):
            if random.random() > 0.85:
                break
            if bbox[4] > 0.5 and bbox[5] > 0.5:
                bbox[0] += 1 / 3 * (bbox[2] - bbox[0])
                bbox[1] += 1 / 3 * (bbox[3] - bbox[1])
                bbox[2] -= 1 / 3 * (bbox[2] - bbox[0])
                bbox[3] -= 1 / 3 * (bbox[3] - bbox[1])
            # print(bbox[0])
            mask[int(bbox[0].item()): int(bbox[2].item()), int(bbox[1].item()):int(bbox[3].item())] = 1
        return mask

    def image_random_mask_from_bbox(self, bboxs, img_shape):
        # print(bboxs)
        mask = torch.ones((img_shape[0], img_shape[1]))
        bbox_index = random.randint(0, bboxs.size(0)-1)
        bbox = bboxs[bbox_index]
        if bbox[4] > 0.5 and bbox[5] > 0.5:
            bbox[0] += 1 / 3 * (bbox[2] - bbox[0])
            bbox[1] += 1 / 3 * (bbox[3] - bbox[1])
            bbox[2] -= 1 / 3 * (bbox[2] - bbox[0])
            bbox[3] -= 1 / 3 * (bbox[3] - bbox[1])
        bbox[0] = bbox[0] * img_shape[0]
        bbox[1] = bbox[1] * img_shape[1]
        bbox[2] = bbox[2] * img_shape[0]
        bbox[3] = bbox[3] * img_shape[1]
        mask[int(bbox[0].item()): int(bbox[2].item()), int(bbox[1].item()):int(bbox[3].item())] = 0
        return mask, bbox_index

    def object_tags_masks(self, indices):
        tags_token_len = sum(int(self.object_token_lens[item])for item in indices)
        tags_token_mask = torch.zeros(len(indices)) # the begin index for each token
        end = 0
        for i, item in enumerate(indices):
            if i == 0:
                end = self.object_token_lens[item]
            else:
                end += int(self.object_token_lens[item])
            tags_token_mask[i] = end
        return tags_token_mask, tags_token_len
        # tags_token_mask = torch.as_tensor([int(self.object_token_lens[item])for item in indices])
        # return tags_token_mask, tags_token_len
        # tags_token_mask = torch.zeros((len(indices), tags_token_len))
        # start = 0
        # for i, item in enumerate(indices):
        #     tags_token_mask[i][start:start + int(self.object_token_lens[item])] = 1
        #     start += int(self.object_token_lens[item])
        # return tags_token_mask, tags_token_len

    def read_object(self, object_path):
        """
        load object features and bounding box localization
        Args:
            object_path(str): absoulte path
            top_k(int): top-k confidence regions
            v(int):  1:  select top-k confidence regions [maybe with same class] 2: select top-k confidence regions with [different classes]
        Returns:
            feat: b x N x [2048+6]; 6 means two points and s_h, s_w
        """
        feat = torch.from_numpy(self.object_dicts[object_path])
        return feat

    def read_bboxs_tags_from_disk(self, object_path, top_k=10, v=1):
        frame1 = np.load(object_path, allow_pickle=True)
        boxes = frame1['bbox']
        # rank features and boxes according to confidence
        confident = frame1['info'].item()['objects_conf']
        condident_indices = np.argsort(confident)[::-1]
        boxes = boxes[condident_indices]
        object_ids = frame1['info'].item()['objects_id']
        object_ids = object_ids[condident_indices]
        if v == 2:
            new_object, unique_indices = np.unique(object_ids, return_index=True)
            # step 2: remove region with same object class
            if len(unique_indices) >= top_k:
                boxes = boxes[unique_indices]
                # print(object_ids, unique_indices, new_object)
                object_ids = object_ids[unique_indices]
        # padding with same elements if not enough
        if boxes.shape[0] < top_k:
            res = top_k - boxes.shape[0]
            boxes = np.pad(boxes, (0, res), 'edge')
            object_ids = np.pad(object_ids, (0, res), 'edge')
        object_tags = ""
        # new_object, unique_indices = np.unique(object_ids, return_index=True)
        # pdb.set_trace()
        # for i in range(10000):
        #     print(new_object)
        for n in range(top_k):
            object_tags += ' ' + (self.classes[object_ids[n]+1])
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
        # print(object_ids.shape, spatial_features.shape)
        return object_tags, object_ids[:top_k], torch.from_numpy(spatial_features)

    def feature_visualize(self, img1, feat_path, outpath):
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
            (scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height),
            axis=1)
        # print(spatial_features)
        feat = torch.cat([torch.from_numpy(features), torch.from_numpy(spatial_features)], dim=1)
        classes = ['__background__']
        with open('utils/objects_vocab.txt', 'r') as f:
            for object in f.readlines():
                classes.append(object.split(',')[0].lower().strip())
        # print(features.shape)
        im = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(im)
        new_boxes = boxes
        for i in range(len(new_boxes)):
            # plot the first bbox
            if i > 0:
                break
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
        plt.savefig(outpath, dpi=150)
        plt.close()
        return outpath

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)
        orig_caption = caption
        # if self.split == 'train':
        #     orig_caption = textaug_eda(orig_caption)
        video_loading = 'no_strict'
        frame_sample = 'rand'
        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']
        try:
            imgs, idxs, vlen, object_index = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample, fix_start=fix_start)
        except:
            if video_loading == 'strict':
                raise ValueError(f'Video loading failed for {video_fp}, video loading for this dataset is strict.')
            else:
                # imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                # imgs = transforms.ToTensor()(imgs).unsqueeze(0)
                print("video is error in: {}".format(rel_fp))
                new_item = random.randint(1, len(self.metadata))
                return self.__getitem__(new_item)

        object_rel_fp, object_fp = self._get_object_path(sample, index=object_index)
        try:
            object_sentence, object_indexs, bboxs = self.read_bboxs_tags_from_disk(object_fp, top_k=20, v=1)
        except OSError:
            # if not found or npz error， return full 1 matrix
            print("object is wrong or not existed in : {}".format(object_fp))
            new_item = random.randint(1, len(self.metadata))
            return self.__getitem__(new_item)
        patch_masks = self.patch_all_masks_from_bbox(bboxs)

        # print("pad_text feature", pad_text_region_features.size())
        # the visualize of first image and

        # # debug and visualization for patch mask
        # vis_img = (imgs[0]*255)
        # print(torch.from_numpy(patch_masks[0]).view(14, 14).unsqueeze(0).unsqueeze(0))
        # m = torch.nn.Upsample([224,224])
        # vis_img = m(vis_img.unsqueeze(0)).squeeze(0)
        # mask = m(torch.from_numpy(patch_masks[0]).view(14, 14).unsqueeze(0).unsqueeze(0))
        # print(mask.size())
        # print(vis_img.size())
        # masked_vis_img = (vis_img * mask.squeeze(0)).permute(1, 2, 0)
        # print(masked_vis_img.size())
        # index = random.randint(1, 10)
        # cv2.imwrite("visualization/{}.png".format(index), masked_vis_img.cpu().detach().numpy())
        # cv2.imwrite("visualization/orig_{}.png".format(index), vis_img.permute(1, 2, 0).cpu().detach().numpy())
        # outPath = "visualization/roi_{}.png".format(index)
        # self.feature_visualize(imgs[0].permute(1, 2, 0).cpu().detach().numpy(), object_fp, outPath)


        if self.text_params['drop_raw_caption']:
            caption = object_sentence
        else:
            caption += object_sentence

        # if self.object_params['input_object_bboxs']:
        #     try:
        #         bboxs = read_bbox_from_disk(object_fp, top_k=10, v=2)
        #     except (KeyError, OSError, RuntimeError) as e:
        #         print("no object available in: {}".format(object_fp))
        #         new_item = random.randint(1, len(self.metadata))
        #         return self.__getitem__(new_item)
        #     # patch_masks = self.patch_mask_from_bbox(bboxs)
        #     patch_masks = self.patch_all_masks_from_bbox(bboxs)
        # multiple patch masks here
        # img_masks = self.image_mask_from_bbox(bboxs, [imgs[0].size(1), imgs[0].size(2)])
        # # print(img_masks.size(), imgs[0].size())
        # imgs[0] = img_masks * imgs[0]
        # img_mask, bbox_index = self.image_random_mask_from_bbox(bboxs, [imgs[0].size(1), imgs[0].size(2)])
        # # img_mask_2, bbox_index_2 = self.image_random_mask_from_bbox(bboxs, [imgs[0].size(1), imgs[0].size(2)])
        # imgs[0] = imgs[0] * img_mask
        # imgs[1:] = imgs[1:] * img_mask_2
        # object len for mask text
        # object_token_len_record = []
        # load object tags
        # if self.text_params['object_tags']:
        #     try:
        #         # object_sentence, object_indexs = read_object_tags_from_disk(object_fp, top_k=10, v=2)
        #     except OSError:
        #         # if not found or npz error， return full 1 matrix
        #         print("object is wrong or not existed in : {}".format(object_fp))
        #         new_item = random.randint(1, len(self.metadata))
        #         return self.__getitem__(new_item)
        #     if self.text_params['drop_raw_caption']:
        #         # if self.dataset_name == 'WebVid':
        #         #     caption = self.read_object_tags(object_rel_fp)
        #         #     # caption = shuffle_object(self.read_object_tags(object_rel_fp))
        #         # else:
        #         #     caption = read_object_tags_from_disk(object_fp)
        #         #     # caption = shuffle_object(read_object_tags_from_disk(object_fp))
        #         caption = object_sentence
        #         # get each token length of object tags
        #         # caption = mask_object_tags(caption, bbox_index)
        #     else:
        #         # if self.dataset_name == 'WebVid':
        #         #     object_sentence = self.read_object_tags(object_rel_fp)
        #         # else:
        #         #     object_sentence = read_object_tags_from_disk(object_fp)
        #         # get each token length of object tags
        #         # object_sentence = mask_object_tags(object_sentence, bbox_index)
        #         # object_sentence = shuffle_object(object_sentence) # shuffle object
        #         # add poseudo and shuffle
        #         # if self.split == 'train':
        #         #     object_sentence = add_pseudo_class(object_sentence)
        #         caption += object_sentence
            # if self.split == 'train':
            #     if self.text_params['text_aug']:
            #         caption = textaug_eda(caption)
        # print(object_indexs, object_sentence)
        # debug for token mask
        object_token_masks, object_token_len = self.object_tags_masks(object_indexs)
        # print(orig_caption)
        # print(caption)
        # print(object_token_masks, object_token_len)
        # print(object_indexs, object_sentence, object_token_masks, object_token_len)
        # patch_masks = 1
        if self.transforms is not None:
            imgs = self.transforms(imgs)
        final = torch.zeros([self.video_params['num_frames'] + 1, 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        # print(len(patch_masks), len(object_token_len_record))
        # print(patch_masks, object_token_len_record)
        data = {'video': final,
                'text': orig_caption, 'pad_text': caption, 'patch_masks': patch_masks,
                'object_token_masks': object_token_masks, 'object_token_len': object_token_len,
                'meta': meta_arr}
        return data



class TextImageDataset(TextVideoDataset):

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)
        #assert os.path.exists(video_fp)
        video_loading = self.video_params.get('loading', 'strict')

        try:
            img = Image.open(video_fp).convert("RGB")
        except:
            if video_loading == 'strict':
                raise ValueError(f'Image loading failed for {video_fp}, image loading for this dataset is strict.')
            else:
                img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))

        # convert to tensor because video transforms don't, expand such that its a 1-frame video.
        img = transforms.ToTensor()(img).unsqueeze(0)
        if self.transforms is not None:
            img = self.transforms(img)
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': img, 'text': caption, 'meta': meta_arr}
        return data


class TextObjectImageDataset(TextObjectVideoDataset):

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        object_rel_fp, object_fp = self._get_object_path(sample)
        caption = self._get_caption(sample)
        orig_caption = caption
        #assert os.path.exists(video_fp)
        video_loading = self.video_params.get('loading', 'strict')
        # load object tags
        if self.text_params['object_tags']:
            if self.text_params['drop_raw_caption']:
                caption = read_object_tags_from_disk(object_fp)
                # caption = shuffle_object(read_object_tags_from_disk(object_fp))
            else:
                object_sentence = read_object_tags_from_disk(object_fp)
                caption += object_sentence
                # object_sentence = shuffle_object(read_object_tags_from_disk(object_fp))
                # if self.text_params['object_aug']:
                #     object_sentence = aug_object(object_sentence)
                # caption = caption + shuffle_object(object_sentence)
        # if self.split == 'train':
        #     if self.text_params['text_aug']:
        #         caption = textaug_eda(orig_caption)
        # print(self.object_params)
        if self.object_params['input_object_bboxs']:
            try:
                bboxs = read_bbox_from_disk(object_fp)
            except (KeyError, OSError, RuntimeError) as e:
                print("no object available in: {}".format(object_fp))
                bboxs = torch.zeros((10, 6))
                bboxs[:, :2] = 0
            # patch_masks = self.patch_mask_from_bbox(bboxs)

        video_loading = 'no_strict'

        try:
            img = Image.open(video_fp).convert("RGB")
        except:
            if video_loading == 'strict':
                raise ValueError(f'Image loading failed for {video_fp}, image loading for this dataset is strict.')
            else:
                img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))

        # convert to tensor because video transforms don't, expand such that its a 1-frame video.
        img = transforms.ToTensor()(img).unsqueeze(0)
        # print(img.size())
        # multiple patch masks here
        img_masks = self.image_mask_from_bbox(bboxs, [img[0].size(1), img[0].size(2)])
        # print(img_masks.size(), img.size())
        mask_img = img_masks * img
        patch_masks = 1
        if self.transforms is not None:
            img = self.transforms(img)
            mask_img = self.transforms(mask_img)
        # print(mask_img - img)
        final = img
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'object_image': mask_img, 'text': orig_caption, 'pad_text': caption,
                'object': patch_masks, 'meta': meta_arr}
        return data

class TextImageTarDataset(TextVideoDataset):
    """
    Borrowed from https://github.com/jotaf98/simple-tar-dataset
    by Joao F. Henriques
    """

    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 tar_fn='images.tar',
                 metadata_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2'
                 ):
        super().__init__(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfms, cut, subsample,
                         sliding_window_stride, reader)

        worker = get_worker_info()
        worker = worker.id if worker else None
        archive = os.path.join(self.data_dir, tar_fn)
        self.tar_obj = {worker: tarfile.open(archive)}
        self.archive = archive
        # store headers of all files and folders by name
        #members = sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name)
        #self.members_by_name = {m.name: m for m in members}

    def get_file(self, name):
        """Read an arbitrary file from the Tar archive.
        Args:
          name (str): File name to retrieve.
        Returns:
          io.BufferedReader: Object used to read the file's content.
        """
        # ensure a unique file handle per worker, in multiprocessing settings
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.archive)

        return self.tar_obj[worker].extractfile(self.members_by_name[name])

    def get_image(self, name):
        """Read an image from the Tar archive, returned as a PIL image or PyTorch tensor.
        Args:
          name (str): File name to retrieve.
        Returns:
          Image or Tensor: The image in PIL format.
        """
        image = Image.open(BytesIO(self.get_file(name).read()))
        return image

    def __getitem__(self, item):
        """Return a single sample.

        Should be overriden by a subclass to support custom data other than images (e.g.
        class labels). The methods get_image/get_file can be used to read from the Tar
        archive, and a dict of files/folders is held in the property members_by_name.
        By default, this simply applies the given transforms or converts the image to
        a tensor if none are specified.
        Args:
          item (int): Index of item.

        Returns:
          Tensor: The image.
        """
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')

        try:
            img = self.get_image(rel_fp)
        except:
            if video_loading == 'strict':
                raise ValueError(f'Image loading failed for {video_fp}, image loading for this dataset is strict.')
            else:
                img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
        img = img.convert('RGB')  # if it's grayscale, convert to RGB
        if self.transforms is not None:  # apply any custom transforms
            img = self.transforms(img)

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': img, 'text': caption, 'meta': meta_arr}
        return data


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


def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    # print("read cv2")
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    object_idxs = sample_frames(8, vlen, sample='uniform')
    average_object_index = int(sum(frame_idxs) / len(frame_idxs))
    average_object_index = min(object_idxs, key=lambda x:abs(x-average_object_index))
    for i, index in enumerate(object_idxs):
        if index == average_object_index:
            object_index = i
    # print(object_index)
    # average_object_index = int(sum(frame_idxs) / len(frame_idxs)) + random.randint(0, 4) - 2
    # average_object_index = min(vlen-1, average_object_index)
    # average_object_index = max(0, average_object_index)
    # print(object_idxs, frame_idxs, average_object_index, vlen)
    frame_idxs.insert(0, average_object_index) # + random.randint(0, 4) - 2) # random image
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
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
    # print(frame_idxs)
    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs, vlen, object_index


def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    average_object_index = int(sum(frame_idxs) / len(frame_idxs)) + random.randint(0, 6) - 3
    average_object_index = min(vlen-1, average_object_index)
    average_object_index = max(0, average_object_index)
    frame_idxs.insert(0, average_object_index)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen

decord.bridge.set_bridge("torch")


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    average_object_index = int(sum(frame_idxs) / len(frame_idxs)) + random.randint(0, 6) - 3
    average_object_index = min(vlen-1, average_object_index)
    average_object_index = max(0, average_object_index)
    frame_idxs.insert(0, average_object_index)
    # print(frame_idxs)
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen


def read_object_tags_from_disk(object_path, top_k=10, v=1, vocab="utils/objects_vocab.txt"):
    frame1 = np.load(object_path, allow_pickle=True)
    classes = ['__background__']
    with open(vocab, 'r') as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())
    # rank features and boxes according to confidence
    confident = frame1['info'].item()['objects_conf']
    condident_indices = np.argsort(confident)[::-1]
    object_ids = frame1['info'].item()['objects_id']
    object_ids = object_ids[condident_indices]
    if v == 2:
        new_object, unique_indices = np.unique(object_ids, return_index=True)
        print(object_ids, unique_indices, new_object)
        # step 2: remove region with same object class
        object_ids = new_object
    # if not enough, pad a background class?
    if object_ids.shape[0] < top_k:
        res = top_k - object_ids.shape[0]
        object_ids = np.pad(object_ids, (0, res), 'edge')
    object_tags = ""
    # new_object, unique_indices = np.unique(object_ids, return_index=True)
    # pdb.set_trace()
    # for i in range(10000):
    #     print(new_object)
    for n in range(top_k):
        object_tags += ' ' + (classes[object_ids[n]+1])
    return object_tags, object_ids[:top_k]


def read_object_from_disk(object_path, top_k=10, v=1):
    """
    load object features and bounding box localization
    Args:
        object_path(str): absoulte path
        top_k(int): top-k confidence regions
        v(int):  1:  select top-k confidence regions [maybe with same class] 2: select top-k confidence regions with [different classes]
    Returns:
        feat: b x N x [2048+6]; 6 means two points and s_h, s_w
    """
    try:
        frame1 = np.load(object_path, allow_pickle=True)
        # print("load success in: {}".format(object_path))
    except OSError:
        # if not found or npz error， return full 1 matrix
        # print("object is wrong or not existed in : {}".format(object_path))
        feat = torch.full((top_k, 2054), 1.0)
        return feat
    features = frame1['x']  # 20 x 2048
    boxes = frame1['bbox']
    # rank features and boxes according to confidence
    confident = frame1['info'].item()['objects_conf']
    condident_indices = np.argsort(confident)[::-1]
    boxes = boxes[condident_indices]
    features = features[condident_indices]
    object_ids = frame1['info'].item()['objects_id']
    if v == 2:
        new_object, unique_indices = np.unique(object_ids, return_index=True)
        # step 2: remove region with same object class
        boxes = boxes[unique_indices]
        features = features[unique_indices]
    # padding with same elements if not enough
    if boxes.shape[0] < top_k:
        res = top_k - boxes.shape[0]
        boxes = np.pad(boxes, (0, res), 'edge')
        features = np.pad(features, (0, res), 'edge')
    boxes = boxes[:top_k, :]
    features = features[:top_k, :]
    # print(boxes, features)
    # scale coordinates to 0-1 as features
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
    # print(feat.size())
    return feat


def read_object_labels_from_disk(object_path, top_k=20, vocab="utils/objects_vocab_fine_grained.txt"):
    try:
        frame1 = np.load(object_path, allow_pickle=True)
    except OSError:
        # if not found or npz error， return full 1 matrix
        object_labels = np.ones(top_k)
        return object_labels
    confident = frame1['info'].item()['objects_conf']
    condident_indices = np.argsort(confident)[::-1]
    object_ids = frame1['info'].item()['objects_id']
    object_ids = object_ids[condident_indices]
    new_object, unique_indices = np.unique(object_ids, return_index=True)
    if len(new_object) < top_k:
        res = top_k - len(new_object)
        new_object = np.pad(new_object, (0, res), 'edge')
    return new_object[:top_k]


def read_bbox_from_pickle(object_path, top_k=10, v=1):
    with open('{}.pickle'.format(object_path), 'rb') as handle:
        object_tags_dicts = pickle.load(handle)
    frame1 = 0
    # frame1 = np.load(object_path, allow_pickle=True)
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


def read_bbox_from_disk(object_path, top_k=10, v=1):
    """
    load object features and bounding box localization
    Args:
        object_path(str): absoulte path
        top_k(int): top-k confidence regions
        v(int):  1:  select top-k confidence regions [maybe with same class] 2: select top-k confidence regions with [different classes]
    Returns:
        feat: b x N x [2048+6]; 6 means two points and s_h, s_w
    """
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


def read_object_info_from_disk(object_path, top_k=10, v=2, vocab="utils/objects_vocab_fine_grained.txt"):
    """
    load 1. object features 2. bounding box localization 3. object tags 4. object classes
    Args:
        object_path(str): absoulte path
        top_k(int): top-k confidence regions
        v(int):  1:  select top-k confidence regions [maybe with same class] 2: select top-k confidence regions with [different classes]
    Returns:
        feat: b x N x [2048+6]; 6 means two points and s_h, s_w
    """
    try:
        frame1 = np.load(object_path, allow_pickle=True)
        # print("load success in: {}".format(object_path))
    except OSError:
        # if not found or npz error， return full 1 matrix
        # print("object is wrong or not existed in : {}".format(object_path))
        feat = torch.full((top_k, 2054), 1.0)
        return feat
    # ========== load features
    features = frame1['x']  # 20 x 2048
    boxes = frame1['bbox']
    # rank features and boxes according to confidence
    confident = frame1['info'].item()['objects_conf']
    condident_indices = np.argsort(confident)[::-1]
    boxes = boxes[condident_indices]
    features = features[condident_indices]
    object_ids = frame1['info'].item()['objects_id']
    if v == 2:
        new_object, unique_indices = np.unique(object_ids, return_index=True)
        # step 2: remove region with same object class
        boxes = boxes[unique_indices]
        features = features[unique_indices]
    # padding with same elements if not enough
    if boxes.shape[0] < top_k:
        res = top_k - boxes.shape[0]
        boxes = np.pad(boxes, (0, res), 'edge')
        features = np.pad(features, (0, res), 'edge')
    boxes = boxes[:top_k, :]
    features = features[:top_k, :]
    # ======== load bbox
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
    # =============== load object tags
    classes = ['__background__']
    with open(vocab, 'r') as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())
    object_tags = ""
    for n in range(0, min(len(new_object), top_k)):
        object_tags += ' ' + (classes[new_object[n] + 1])
    # ================ load
    object_label = new_object[:top_k]
    return features, spatial_features, object_tags, object_label




video_reader = {
    'av': read_frames_av,
    'cv2': read_frames_cv2,
    'decord': read_frames_decord
}
