# from base.base_dataset import TextObjectImageDataset
from OATrans.base.base_dataset_region_mem import TextObjectImageDataset
import pandas as pd
import os


class ConceptualCaptions3M(TextObjectImageDataset):
    """
    Conceptual Captions dataset. Split files are specific to my download regime.
    """

    def _load_metadata(self):
        # download specific
        metadata_dir = './meta_data'
        split_files = {
            'train': 'cc3m_training_success_full.tsv',
            'val': 'cc3m_validation_success_full.tsv',            # there is no test
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        # elif self.split == 'val':
        #     metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        self.metadata = metadata

    def _get_video_path(self, sample):
        # conceptual captions uses this hashing to create the filename
        rel_dir = 'training'
        if self.split != 'train':
            rel_dir = 'validation'
        rel_fp = os.path.join(rel_dir, sample[1])
        #rel_fp = os.path.join(rel_dir, str(zlib.crc32(sample['thumbnailUrl'].encode('utf-8')) & 0xffffffff))
        return os.path.join(self.data_dir, rel_fp), rel_fp

    def _get_caption(self, sample):
        return sample[0]
        #return sample['caption']

    def _get_object_path(self, sample):
        """
        get the object npy path
        Args:
            sample (dict):
        Returns:
            abs path
        """
        # pre = sample[1].split('_')[0]
        # pre = pre.zfill(7)
        # rel_object_fp = os.path.join(pre[:4], sample[1])
        # rel_object_fp = os.path.join(pre[:4], sample[1] + '_1.npz')
        rel_object_fp = sample[1]
        full_object_fp = os.path.join(self.object_dir, self.split, rel_object_fp)
        return os.path.join(self.split, rel_object_fp), full_object_fp