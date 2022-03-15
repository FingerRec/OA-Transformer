from model.text_model import text_encode_init
from model.vision_model import vision_encode_init
from data_preprocess import vision_img_preprocess, clip_img_preprocess
import pandas as pd
from visualize import cross_attention_visualize
import parse_config as parse_config
import model.vision_models.clip as clip


csv_file = "data/meta_data/cc3m_training_success_full.tsv"
# out_dir = 'output/featmap/'
model_se = 'frozen'  # 'frozen' or 'clip'
# out_dir = 'output/featmap/{}/'.format(model_se)
out_dir = 'output/cross_featmap/cc3m/{}_attn/'.format(model_se)
video_root = 'CC3M/training/'
metadata = pd.read_csv(csv_file, sep='\t')
text_model = text_encode_init(model_name=model_se)
img_model, preprocess = vision_encode_init(model_name=model_se)

count = 0
for item in range(len(metadata)):
    sample = metadata.iloc[item]
    video_src = video_root + sample[1]
    caption = sample[0]
    if model_se == 'clip':
        img = clip_img_preprocess(video_src, preprocess)
    else:
        img = vision_img_preprocess(video_src)
    print(img.size())
    img_patch_embedding = img_model(img)
    if model_se == 'clip':
        img = img.unsqueeze(0)
    if model_se == 'clip':
        text_token = text_model(clip.tokenize(caption).cuda())
    else:
        text_token = text_model(caption)
    # print(img_patch_embedding.size())
    if model_se == 'clip':
        img = img.float()
    cross_attention_visualize(img_patch_embedding, img[0], caption, text_token, text_model, model_name=model_se,
                                    name=out_dir + str(item), v=1)
    count += 1
    if count > 500:
        break


