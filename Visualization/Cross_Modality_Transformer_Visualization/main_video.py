from model.text_model import text_encode_init
from model.vision_model import vision_encode_init
from data_preprocess import vision_preprocess
import pandas as pd
from visualize import cross_attention_visualize
import parse_config as parse_config


csv_file = "data/webvid_validation_success_full.tsv"
# out_dir = 'output/featmap/'
out_dir = 'output/cross_featmap/'
video_root = 'WebVid/val/'
metadata = pd.read_csv(csv_file, sep='\t')
text_model = text_encode_init()
video_model = vision_encode_init()

count = 0
for item in range(len(metadata)):
    sample = metadata.iloc[item]
    video_src = video_root + sample[1] + '.mp4'
    caption = sample[0]
    # print(video_src)
    video = vision_preprocess(video_src)
    # print(video.size())
    video_patch_embedding = video_model(video)
    print(caption)
    text_token = text_model(caption)
    # print(video_patch_embedding.size())
    # print(text_token.size())
    sim = cross_attention_visualize(video_patch_embedding, video[0], caption, text_token, text_model, name=out_dir + str(item))

    count += 1
    if count > 100:
        break

