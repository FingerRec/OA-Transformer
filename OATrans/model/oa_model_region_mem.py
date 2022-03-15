import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from OATrans.base import BaseModel
from OATrans.utils.util import state_dict_data_parallel_fix
from transformers import AutoModel
import torch
# from model.video_transformer import SpaceTimeTransformer
from OATrans.model.oa_video_transformer_region import SpaceTimeTransformer


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)



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
        print(self.text_model)
        self.text_model.train()
        pretrained = video_params['pretrained']
        if video_params['model'] in ["SpaceTimeTransformer", "SpaceTimeObjectTransformer"]:
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            if arch_config == 'base_patch16_224':
                vit_model = torch.load("pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                if video_params['model'] == "SpaceTimeTransformer":
                    model = SpaceTimeTransformer(num_frames=num_frames,
                                                time_init=time_init,
                                                attention_style=attention_style)
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
            self.video_model.fc = nn.Identity()
        elif video_params['model'] == "":
            print("no vision model available!")
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")
        if projection == 'minimal':
            txt_proj = nn.Sequential(nn.ReLU(),
                                         nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                         )
            txt_proj_2 = nn.Sequential(nn.ReLU(),
                                         nn.Linear(512, projection_dim),
                                         )
            if video_params['model'] != "":
                vid_proj = nn.Sequential(
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
        self.txt_proj_2 = txt_proj_2
        if video_params['model'] != "":
            self.vid_proj = vid_proj

        self.txt_proj.apply(init_weights)
        self.vid_proj.apply(init_weights)

        self.txt_proj_2.apply(init_weights)

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=False)
        self.sigmod = nn.Sigmoid()

    def set_device(self, device):
        self.device = device

    def forward(self, data, aug=False, return_embeds=True):
        text_data = data['text']
        video_data = data['video']
        text_region_embedding = data['text_region_embedding']
        text_embeddings = self.compute_text(text_data)
        # print(video_data.size())
        video_data = video_data.view(video_data.size(0) * 2, -1, video_data.size(2), video_data.size(3),
                                     video_data.size(4))
        vision_embeddings, vision_region_feature = self.compute_video(video_data)
        object_image_embeddings, object_region_feature = vision_embeddings[0::2].contiguous(), vision_region_feature[
                                                                                               0::2].contiguous()
        video_embeddings, video_region_feature = vision_embeddings[1::2].contiguous(), vision_region_feature[
                                                                                       1::2].contiguous()
        text_region_embedding = self.txt_proj_2(text_region_embedding)
        video_embeddings = (video_embeddings + torch.mean(video_region_feature, dim=1)) / 2
        # print(text_embeddings.size(), text_region_embedding.size())
        # text_embeddings = (text_embeddings + torch.mean(text_region_embedding, dim=1)) / 2  # add region info into sentence
        region_sim = self.compute_region_sim(object_region_feature, text_region_embedding)
        return text_embeddings, video_embeddings, region_sim

    def compute_text(self, text_data, pad=False):
        if self.text_params['model'].split('/')[-1].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].split('/')[-1].startswith('distilbert'):
            text_embeddings_all = self.text_model(**text_data).last_hidden_state
            text_embeddings = text_embeddings_all[:, 0, :]
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings.float())
        return text_embeddings

    def compute_object(self, object_data):
        object_embeddings = self.object_model(object_data)
        return object_embeddings

    def compute_video(self, video_data):
        video_embeddings, region_feature = self.video_model(video_data)
        video_embeddings_proj = self.vid_proj(video_embeddings)
        region_feature_proj = self.vid_proj(region_feature)
        return video_embeddings_proj, region_feature_proj

    def compute_region_sim(self, video_feats, text_feats):
        # print(video_feats.size(), text_feats.size())
        # sim = einsum('b k f, b n f -> b k n', F.normalize(text_feats, dim=2), F.normalize(video_feats, dim=2))
        sim = einsum('b k f, b n f -> b k n', text_feats, video_feats)
        return self.sigmod(sim)

    def compute_video_object_joint(self, video_data, object_data):
        if self.video_params['two_outputs']:
            pad_joint_embeddings, joint_embeddings  = self.video_model(video_data, object_data)
            joint_embeddings = self.joint_proj(joint_embeddings)
            pad_joint_embeddings = self.joint_proj(pad_joint_embeddings)
            return joint_embeddings, pad_joint_embeddings
        else:
            joint_embeddings = self.video_model(video_data, object_data)
            # print(joint_embeddings.size())
            joint_embeddings = self.joint_proj(joint_embeddings)
            return joint_embeddings

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
