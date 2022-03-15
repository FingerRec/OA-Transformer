# from base.base_dataset import TextObjectVideoDataset
from OATrans.base.base_dataset_region_mem import TextObjectVideoDataset
# from base.base_dataset_region_single import TextObjectVideoDataset
# from base.base_dataset_region_mem_bk import TextObjectVideoDataset
import pandas as pd
import os


class WebVidObject(TextObjectVideoDataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def _load_metadata(self):
        #metadata_dir = os.path.join(self.metadata_dir, 'meta_data')
        metadata_dir = './meta_data'
        split_files = {            
            'train': 'webvid_training_success_full.tsv',
            # 'train': 'webvid_1_of_10_training_success_full.tsv',
            # 'train': 'webvid_validation_success_full.tsv',
            'val': 'webvid_validation_success_full.tsv',            # there is no test
        }

        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        # elif self.split == 'val':
        #     metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        #metadata['caption'] = metadata['name']
        #del metadata['name']
        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary
        #self.metadata.dropna(inplace=True)
        #self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        rel_video_fp = sample[1] + '.mp4'
        #rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, self.split, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample[0]

    def _get_object_path(self, sample):
        """
        get the object npy path
        Args:
            sample (dict):
        Returns:
            abs path
        """
        # rel_object_fp = sample[1] + '.pickle'
        rel_object_fp = sample[1]  # + '.pickle'
        full_object_fp = os.path.join(self.object_dir, self.split, rel_object_fp)
        return rel_object_fp, full_object_fp