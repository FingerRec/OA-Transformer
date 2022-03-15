import random

import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, \
    BertTokenizer, T5EncoderModel
import torch
import timm
from model.oa_video_transformer_global_local import SpaceTimeTransformer
import pdb
from torch import einsum
import cv2

class FrozenInTime(BaseModel):
    def __init__(self,
                 video_params,
                 object_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()
        self.video_params = video_params
        self.text_params = text_params
        self.object_params = object_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()
        #pdb.set_trace()
        if object_params['model'] == 'mlp':
            # print("*"*200)
            self.object_model = SimpleMLP()
        elif object_params['model'] == 'ObjectTransformer':
            self.object_model = load_clip_pt_weight(ObjectTransformer(input_dim=2054, region_nums=10, output_dim=256))
        elif object_params['model'] == "":
            self.object_model = None
        else:
            raise NotImplementedError("only support mlp and ObjectTransformer now")
        # self.object_model = SimpleMLP(input_dim=2054, hidd_dim=256, object_num=10, out_dim=256)
        # pdb.set_trace()
        pretrained = video_params['pretrained']
        if video_params['model'] in ["SpaceTimeTransformer", "SpaceTimeObjectTransformer"]:
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            vit_init = video_params.get('vit_init', 'imagenet-21k')
            modality_token = video_params.get('modality_token', False)
            two_outputs = video_params.get('two_outputs', False)
            if arch_config == 'base_patch16_224':
                # vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                # vit_model = torch.jit.load("pretrained/jx_vit_base_p16_224-80ecf9dd.pth")
                # vit_model = torch.jit.load("pretrained/ViT-B-16.pt")
                vit_model = torch.load("pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                if video_params['model'] == "SpaceTimeTransformer":
                    model = SpaceTimeTransformer(num_frames=num_frames,
                                                time_init=time_init,
                                                attention_style=attention_style)
                elif video_params['model'] == "SpaceTimeObjectTransformer":
                    model = SpaceTimeObjectTransformer(num_frames=num_frames,
                                                 time_init=time_init,
                                                 attention_style=attention_style,
                                                 modality_token=modality_token,
                                                 two_outputs=two_outputs)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                vit_checkpoint = vit_model
                # vit_checkpoint = vit_model.state_dict()
                model.load_state_dict(vit_checkpoint, strict=False)
            self.video_model = model
            # for backwards compatibility (old models)
            self.video_model.fc = nn.Identity()
        elif video_params['model'] == "clip_vit":
            model = VisionTransformer()
            ftr_dim = 768
            if pretrained:
                print("load clip vit state dict from local")
                pretrained_state = torch.jit.load('pretrained/ViT-B-16.pt').visual.state_dict()
                model.load_state_dict(pretrained_state)
            self.video_model = model

        elif video_params['model'] == "":
            print("no vision model available!")
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")


        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )
            txt_proj_2 = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )
            if video_params['model'] != "":
                vid_proj = nn.Sequential(
                    nn.Linear(ftr_dim, projection_dim)
                )
                vid_proj_2 = nn.Sequential(
                    nn.Linear(ftr_dim, projection_dim)
                )
            if video_params["two_outputs"]:
                ftr_dim = 768
                joint_proj = nn.Sequential(
                    nn.Linear(ftr_dim, projection_dim)
                )
            if object_params['model'] != "":
                ftr_dim = 768
                joint_proj = nn.Sequential(
                    nn.Linear(ftr_dim, projection_dim)
                )
        elif projection != '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
            if object_params['model'] == "":
                joint_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.text_local_proj = txt_proj_2
        if video_params['model'] != "":
            self.vid_proj = vid_proj
            self.vid_local_proj = vid_proj_2
        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=False)
        # model
        self.cross_model = CrossModalityFusion()
        self.cross_model.train()

    def set_device(self, device):
        self.device = device

    def forward(self, data, return_embeds=True):
        text_data = data['text']
        video_data = data['video']
        text_embeddings, text_region_features = self.compute_text(text_data)
        pad_text_data = data['pad_text']
        pad_text_embeddings, pad_text_region_features = self.compute_text(pad_text_data)
        video_data = video_data.view(video_data.size(0)*2, -1, video_data.size(2), video_data.size(3), video_data.size(4))
        vision_embeddings, vision_region_feature = self.compute_video(video_data)
        # print(vision_embeddings.size(), vision_embeddings.size()) # 196 x 768
        object_image_embeddings, object_region_feature = vision_embeddings[0::2].contiguous(), vision_region_feature[0::2].contiguous()
        video_embeddings, video_region_feature = vision_embeddings[1::2].contiguous(), vision_region_feature[1::2].contiguous()
        # print(video_embeddings.size(), object_image_embeddings.size())
        # # video_embeddings = video_embeddings/2 + object_image_embeddings/2
        # # return text_embeddings, pad_text_embeddings, video_embeddings, object_image_embeddings
        # f_q = self.cross_model(text_region_features, video_region_feature).contiguous()
        # # k = self.cross_model(pad_text_embeddings, video_embeddings)
        # f_k = self.cross_model(pad_text_region_features, video_region_feature).contiguous()
        # # n = self.cross_model(pad_text_embeddings, object_image_embeddings)
        # if random.random() < 0.5:
        #     f_n = self.cross_model(text_region_features.flip(0), video_region_feature).contiguous()
        # else:
        #     f_n = self.cross_model(text_region_features, video_region_feature.flip(0)).contiguous()
        # q = video_embeddings
        # k = object_image_embeddings
        # n = video_embeddings.flip(0)

        # fine-grained match
        patch_masks = data['patch_masks'].float()
        # print(patch_masks.size(), object_region_feature.size()) # 24x20x196, 24x196x768
        region_feat = einsum('b o l, b l c -> b o c', patch_masks, object_region_feature)
        # print(region_feat.size()) # 24 x 20 x 768
        object_token_masks = data['object_token_masks']
        object_token_len = data['object_token_len']
        # print(object_token_masks.size(0), object_token_masks.size(1), pad_text_region_features.size(1))
        tags_masks = torch.zeros(object_token_masks.size(0), object_token_masks.size(1), pad_text_region_features.size(1)).float().to(self.device) # b x o x l
        B, O, L = tags_masks.size()
        # print("padded length is: ", pad_text_data['input_ids'].size(1))
        # print("is: ", torch.sum(text_data['attention_mask'][0]).item(), torch.sum(pad_text_data['attention_mask'][0]).item(), object_token_len[0].item())
        # 3.g. 14 21 7
        # only place 1 for place with object word token
        for j in range(B):
            # k is the specific object
            start = 0
            for k in range(object_token_masks.size(1)):
                # print(text_data['attention_mask'][j][k])
                tags_masks[j][k][torch.sum(text_data['attention_mask'][j])-1+start:
                                 torch.sum(text_data['attention_mask'][j])-1+ int(object_token_masks[j][k])] = 1
                start = int(object_token_masks[j][k])
        # print(tags_masks[0][4])
        # print(text_region_features.size()) # the word will padded to max length
        # print(pad_text_data)
        tags_feat = einsum('b o l, b l c -> b o c', tags_masks, pad_text_region_features)
        # ======== then projected to same embedding space
        region_feat = self.vid_local_proj(region_feat)
        tags_feat = self.text_local_proj(tags_feat)
        # print(region_feat.size())
        # print(tags_feat.size())
        return text_embeddings, pad_text_embeddings, video_embeddings, object_image_embeddings, \
               [text_region_features, pad_text_region_features, video_region_feature, object_region_feature,
                region_feat, tags_feat]

    def compute_text(self, text_data):
        if self.text_params['model'].split('/')[-1].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].split('/')[-1].startswith('distilbert'):
            text_embeddings_all = self.text_model(**text_data).last_hidden_state
            # print(text_embeddings_all.size())
            text_embeddings = text_embeddings_all[:, 0, :] + torch.mean(text_embeddings_all[:, 1:, :], dim=1)
            text_region_features = text_embeddings_all # [:, 1:, :] # notice the first is padded 101, the last is padded 102
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings, text_region_features

    def compute_object(self, object_data):
        object_embeddings = self.object_model(object_data)
        return object_embeddings

    def compute_video(self, video_data):
        video_embeddings, region_feature = self.video_model(video_data)
        video_embeddings_proj = self.vid_proj(video_embeddings)
        return video_embeddings_proj, region_feature

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')
        return new_state_dict


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


if __name__ == "__main__":
    pass
