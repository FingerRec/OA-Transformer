from data_preprocess import mask_vision_preprocess
import pandas as pd
import os
from patch_mask import visualize_mask
from utils.read_bboxs import read_bbox_from_pickle


csv_file = "data/webvid_validation_success_full.tsv"
# out_dir = 'output/featmap/'
out_dir = 'output/mask_object_visualization/'
video_root = 'WebVid/val/'
metadata = pd.read_csv(csv_file, sep='\t')
features_root = 'WebVid/8_frame_object'



count = 0
for item in range(len(metadata)):
    sample = metadata.iloc[item]
    video_src = video_root + sample[1] + '.mp4'
    video = mask_vision_preprocess(video_src)
    object_bboxs = []
    for i in range(3):
        rel_object_fp = os.path.join(sample[1], '{}.npz'.format(i))
        full_object_fp = os.path.join(features_root, 'val', rel_object_fp)
        object_bboxs.append(read_bbox_from_pickle(full_object_fp))
    # print(video_src)
    out_name = out_dir + str(item)
    visualize_mask(video, object_bboxs, out_name)
    count += 1
    if count > 100:
        break

