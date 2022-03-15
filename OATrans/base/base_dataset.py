import cv2
import av
import os
import numpy as np
import torch
import random
from PIL import Image
from abc import abstractmethod
import tarfile
from io import BytesIO
from torchvision import transforms
from torch.utils.data import Dataset, get_worker_info
import decord


# import nltk
# nltk.data.path.append("pretrained/nltk_data")
# from textaugment import EDA

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
                    transforms.Resize(size=(238, 238)),
                    transforms.RandomCrop(size=(224, 224)),
                    # add for data augmentation
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    # transforms.RandomApply([base_augmentation.GaussianBlur([.1, 2.])], p=0.5), # PIL
                    # transforms.ToTensor(), # PIL
                    # transforms.RandomResizedCrop(size=(224, 224)),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(size=(238, 238)),
                    transforms.CenterCrop(size=(224, 224)),
                    # transforms.RandomResizedCrop(size=(224, 224)),
                    # transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

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

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        # print("233")
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)
        video_loading = 'no_strict'
        frame_sample = 'rand'
        # frame_sample = 'uniform'
        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']

        try:
            imgs, idxs, vlen = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample, fix_start=fix_start)
        except:
            print("no videos loaded in: {}".format(video_fp))
            if video_loading == 'strict':
                raise ValueError(f'Video loading failed for {video_fp}, video loading for this dataset is strict.')
            else:
                print("video is error in: {}".format(rel_fp))
                new_item = random.randint(1, len(self.metadata))
                return self.__getitem__(new_item)
        if self.transforms is not None:
            imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr}
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
        # prompt = "this is a image of "
        prompt = ""
        caption = prompt + self._get_caption(sample)
        orig_caption = caption
        #assert os.path.exists(video_fp)
        video_loading = self.video_params.get('loading', 'strict')
        # load object tags
        if self.text_params['object_tags']:
            if self.text_params['drop_raw_caption']:
                caption = read_object_tags_from_disk(object_fp)
            else:
                object_sentence = read_object_tags_from_disk(object_fp)
                # if self.text_params['object_aug']:
                #     # object_sentence = aug_object_del(object_sentence)
                #     object_sentence = add_pseudo_class(object_sentence)
                # #     object_sentence = aug_object(object_sentence)
                caption = caption + object_sentence
            # if self.split == 'train':
            #     if self.text_params['text_aug']:
            #         caption = textaug_eda(caption)
        if self.object_params['input_objects']:
            try:
                object = read_object_from_disk(object_fp)
            except (RuntimeError, zipfile.BadZipFile) as e:
                object = torch.full((10, 2054), 1.0)
        if self.object_params['input_object_bboxs']:
            try:
                object = read_bbox_from_disk(object_fp)
            except (KeyError, OSError, RuntimeError) as e:
                print("")
                object = torch.zeros((10, 6))
        if self.object_params['pseudo_labels']:
            try:
                pseudo_labels = self.read_pseudo_label(object_fp)
            except (KeyError, RuntimeError) as e:
                print("{} not found label".format(object_fp))
                pseudo_labels = torch.zeros(1601)
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
        if self.transforms is not None:
            img = self.transforms(img)
        final = img
        # data['object_pseduo_label'] = 1
        if self.object_params['input_objects']:
            meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
            if self.text_params['object_tags']:
                data = {'video': final, 'text': orig_caption, 'pad_text': caption, 'object': object, 'meta': meta_arr}
            else:
                data = {'video': final, 'text': orig_caption, 'object': object, 'meta': meta_arr}
        else:
            meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
            if self.text_params['object_tags']:
                data = {'video': final, 'text': orig_caption, 'pad_text': caption, 'meta': meta_arr}
            else:
                data = {'video': final, 'text': orig_caption, 'meta': meta_arr}
        if self.object_params['pseudo_labels']:
            data['pseudo_labels'] = pseudo_labels
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


def read_clip_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    cap.release()
    return frames


def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
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
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs, vlen


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
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
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


def read_object_tags_from_disk(object_path, top_k=5, vocab="utils/objects_vocab.txt"):
    # print(object_path)
    try:
        # print("load success in: {}".format(object_path))
        frame1 = np.load(object_path, allow_pickle=True)
    except OSError:
        # if not found or npz error， return full 1 matrix
        print("object is wrong or not existed in : {}".format(object_path))
        object_tags = ""
        return object_tags
    classes = ['__background__']
    with open(vocab, 'r') as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())
    # rank features and boxes according to confidence
    confident = frame1['info'].item()['objects_conf']
    condident_indices = np.argsort(confident)[::-1]
    object_ids = frame1['info'].item()['objects_id']
    object_ids = object_ids[condident_indices]
    object_tags = ""
    new_object, unique_indices = np.unique(object_ids, return_index=True)
    # pdb.set_trace()
    # for i in range(10000):
    #     print(new_object)
    for n in range(0, min(len(new_object), top_k)):
        object_tags += ' ' + (classes[new_object[n]+1])
    return object_tags


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


def read_object_labels_from_disk(object_path, top_k=30, vocab="utils/objects_vocab_fine_grained.txt"):
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