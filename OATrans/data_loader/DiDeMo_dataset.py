from OATrans.base.base_dataset import TextObjectVideoDataset
import pandas as pd
import os


class DiDeMo(TextObjectVideoDataset):
    def _load_metadata(self):
        metadata_dir = './meta_data'
        split_files = {
            'train': 'DiDeMo_train.tsv',
            'val': 'DiDeMo_val.tsv',            # there is no test
            'test': 'DiDeMo_test.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))

    def _get_video_path(self, sample):
        rel_video_fp = sample[1]
        #rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        # print(full_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        # print(sample[0].split(',')[0])
        # return sample[0].split(',')[0]
        return sample[0] # .split(',')[0]

    def _get_object_path(self, sample, index=0):
        """
        get the object npy path
        Args:
            sample (dict):
        Returns:
            abs path
        """
        rel_object_fp = os.path.join(sample[1], '1.npz')
        full_object_fp = os.path.join(self.object_dir, self.split, rel_object_fp)
        return os.path.join(self.split, rel_object_fp), full_object_fp