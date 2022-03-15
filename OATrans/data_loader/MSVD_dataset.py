import random

from OATrans.base.base_dataset import TextObjectVideoDataset
import pandas as pd
import os


class MSVD(TextObjectVideoDataset):
    def _load_metadata(self):
        metadata_dir = './meta_data'
        split_files = {
            'train': 'MSVD_train.tsv',
            # 'val': 'MSVD_val.tsv',            # there is no test
            'val': 'MSVD_test.tsv',  # direct output test result
            # 'val': 'MSVD_split_test.tsv',
            # 'test': 'MSVD_split_test.tsv'
            'test': 'MSVD_test.tsv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        self.metadata = metadata
        print("load split {}, {} samples".format(self.split, len(metadata)))

    def _get_video_path(self, sample):
        rel_video_fp = sample[1] + '.avi'
        #rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        # print(full_video_fp)
        return full_video_fp, rel_video_fp

        # multiple sentence
    def _get_caption(self, sample):
        # print(sample[0].split(',')[0])
        if self.split == 'train':
            words = sample[0].split(',')
            num_word = len(words)
            index = random.randint(0, num_word-1)
            caption = words[index]
        else:
            # caption = sample[0]
            words = sample[0].split(',')
            num_word = len(words)
            index = random.randint(0, num_word-1)
            caption = words[index]
        # caption = None
        # if self.split == 'train':
        #     indexs = sorted(random.sample(range(0, num_word-1), 5))
        #     caption = ' '.join(words[item] for item in indexs)
        # else:
        #     caption = ' '.join(words[item] for item in range(0, 5))
        return caption

    def _get_object_path(self, sample, index=1):
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