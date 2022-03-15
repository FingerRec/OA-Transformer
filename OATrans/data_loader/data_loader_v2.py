from OATrans.base import BaseDataLoaderExplicitSplit, DistBaseDataLoaderExplicitSplit, MultiDistBaseDataLoaderExplicitSplit, BaseMultiDataLoader
from OATrans.data_loader.data_loader import init_transform_dict
from OATrans.data_loader.data_loader import ConceptualCaptions3M
from OATrans.data_loader.data_loader import MSRVTT
from OATrans.data_loader.data_loader import LSMDC
from OATrans.data_loader.data_loader import WebVidObject
from OATrans.data_loader.data_loader import MSVD
from OATrans.data_loader.data_loader import DiDeMo


def dataset_loader(dataset_name,
                   text_params,
                   video_params,
                   data_dir,
                   metadata_dir=None,
                   split='train',
                   tsfms=None,
                   cut=None,
                   subsample=1,
                   sliding_window_stride=-1,
                   reader='cv2'):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "SomethingSomethingV2":
        dataset = SomethingSomethingV2(**kwargs)
    elif dataset_name == "WebVid":
        dataset = WebVid(**kwargs)
    elif dataset_name == "ConceptualCaptions3M":
        dataset = ConceptualCaptions3M(**kwargs)
    elif dataset_name == "ConceptualCaptions12M":
        dataset = ConceptualCaptions12M(**kwargs)
    elif dataset_name == "LSMDC":
        dataset = LSMDC(**kwargs)
    elif dataset_name == "COCOCaptions":
        dataset = COCOCaptions(**kwargs)
    elif dataset_name == "MSVD":
        dataset = MSVD(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


def dataset_object_loader(dataset_name,
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
                   reader='cv2'):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        object_params=object_params,
        video_params=video_params,
        data_dir=data_dir,
        object_dir=object_dir,
        metadata_dir=metadata_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "WebVid":
        dataset = WebVidObject(**kwargs)
    elif dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "ConceptualCaptions3M":
        dataset = ConceptualCaptions3M(**kwargs)
    elif dataset_name == "MSVD":
        dataset = MSVD(**kwargs)
    elif dataset_name == "DiDeMo":
        dataset = DiDeMo(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset

class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        #        if split != 'train':
        #            shuffle = False

        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name


class DistTextVideoDataLoader(DistBaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        #        if split != 'train':
        #            shuffle = False
        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name


class MultiDistTextObjectVideoDataLoader(MultiDistBaseDataLoaderExplicitSplit):
    def __init__(self,
                 args,
                 dataset_name,
                 text_params,
                 object_params,
                 video_params,
                 data_dir,
                 object_dir,
                 metadata_dir=None,
                 split='train',
                 tsfm_params=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 batch_size=8,
                 num_workers=8,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        tsfm_dict = init_transform_dict(**tsfm_params)
        tsfm = tsfm_dict[split]
        dataset = dataset_object_loader(dataset_name, text_params, object_params, video_params, data_dir, object_dir, metadata_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader)
        #        if split != 'train':
        #            shuffle = False
        # print(batch_size)
        # print(num_workers)
        super().__init__(args, dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name


class TextVideoMultiDataLoader(BaseMultiDataLoader):
    # TODO: figure out neat way to have N data_loaders
    # TODO: also add N weighted sampler
    def __init__(self, data_loader1, data_loader2):
        # get class from "type" in dict
        dls_cfg = [data_loader1, data_loader2]
        dls = []
        for dcfg in dls_cfg:
            dl = globals()[dcfg['type']](**dcfg['args'])
            dls.append(dl)
        super().__init__(dls)
