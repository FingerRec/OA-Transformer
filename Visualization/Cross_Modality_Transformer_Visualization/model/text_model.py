import torch.nn as nn
from transformers import AutoModel
import transformers
import torch
import model.vision_models.clip as clip

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained('pretrained/distilbert-base-uncased')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('pretrained/distilbert-base-uncased',
                                                               TOKENIZERS_PARALLELISM=False)
        self.device = "cuda:0"
        self.txt_proj =  nn.Sequential(nn.ReLU(),
                                     nn.Linear(768, 256),
                                     )
    def token_of_word(self, word):
        token = self.tokenizer(word, return_tensors='pt', padding=True,
                                          truncation=True)
        return token

    def forward(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer(x, return_tensors='pt', padding=True,
                                          truncation=True)
        x = {key: val.to(self.device) for key, val in x.items()}
        text_embeddings_all = self.text_model(**x).last_hidden_state
        # print(text_embeddings_all.size()) # batch_size, sequence_length, hidden_size
        # text_embeddings = text_embeddings_all[:, 0, :]
        text_embeddings = text_embeddings_all
        # print(text_embeddings.size())
        return self.txt_proj(text_embeddings)
        # return text_embeddings

def weight_transform(model_dict, pretrain_dict):
    '''
    :return:
    '''
    weight_dict = {k[7:]:v for k, v in pretrain_dict.items() if k[7:] in model_dict and k[:7] == 'module.'}
    # for k, v in pretrain_dict.items():
    #     print(k[7:])
    # #     pdb.set_trace()
    for k, v in pretrain_dict.items():
        if k[:14] == 'module.txt_proj':
            weight_dict[k[7:]] = v
    for k, v in weight_dict.items():
        print("load: {}".format(k))
    # print(weight_dict)
    model_dict.update(weight_dict)
    return model_dict

def load_pt_weight(model):
    checkpoint = torch.load("pretrained/cc-webvid2m-4f_stformer_b_16_224.pth.tar", map_location="cpu")
    pretrained_state = checkpoint['state_dict']
    model_state = model.state_dict()
    # for k , v in model_state.items():
    #     print(k)
    model.load_state_dict(weight_transform(model_state, pretrained_state))
    return model


def text_encode_init(model_name='frozen'):
    if model_name == 'clip':
        full_model, preprocess = clip.load("pretrained/ViT-B-16.pt")
        model = full_model.encode_text
    else:
        model = TextEncoder()
        load_pt_weight(model)
        model = model.cuda()
    return model