import torch
from model.vision_models.frozen import SpaceTimeTransformer
import model.vision_models.clip as clip


def weight_transform(model_dict, pretrain_dict):
    '''
    :return:
    '''
    weight_dict = {k[19:]:v for k, v in pretrain_dict.items() if k[19:] in model_dict and k[:19] == 'module.video_model.'}
    for k, v in pretrain_dict.items():
        print(k[19:])
    #     pdb.set_trace()
    for k, v in pretrain_dict.items():
        if k[:15] == 'module.vid_proj':
            weight_dict[k[7:]] = v
    for k, v in weight_dict.items():
        print("load: {}".format(k))
    # print(weight_dict)
    model_dict.update(weight_dict)
    return model_dict


def load_pt_weight(model):
    """
    load the object transformer weight from clip vision transformer
    notice some of have failed
    Args:
        model ():

    Returns:

    """
    checkpoint = torch.load("pretrained/cc-webvid2m-4f_stformer_b_16_224.pth.tar", map_location="cpu")
    pretrained_state = checkpoint['state_dict']
    # model.load_state_dict(vit_checkpoint, strict=False)
    # pretrain_model = torch.jit.load('pretrained/ViT-B-16.pt')
    # pretrained_state = pretrain_model.state_dict()
    model_state = model.state_dict()
    # for k, v in model_state.items():
    #     print(k)
    model.load_state_dict(weight_transform(model_state, pretrained_state))
    return model


def vision_encode_init(model_name="frozen"):
    # frozen
    preprocess = None
    if model_name == 'clip':
        full_model, preprocess = clip.load("pretrained/ViT-B-16.pt")
        model = full_model.visual
    elif model_name == 'frozen':
        model = SpaceTimeTransformer()
        load_pt_weight(model)
    else:
        print("not support")
    model = model.cuda()
    return model, preprocess