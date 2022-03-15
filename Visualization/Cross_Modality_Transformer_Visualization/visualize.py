import torch
import numpy as np
from scipy.ndimage import zoom
import os
import cv2
import torch.nn.functional as F
from torch import einsum
import nltk
nltk.data.path.append("pretrained/nltk_data")
import model.vision_models.clip as clip


def check_nouns(words):
    is_noun = lambda pos: pos[:2] == 'NN'
    # do the nlp stuff
    tokenized = nltk.word_tokenize(words)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    if len(nouns) > 0:
        return True
    else:
        return False


def cam_calculate(model_ft, vid):
    # get predictions, last convolution output and the weights of the prediction layer
    # i3d is two layer fc, need to modify here
    predictions, layerout = model_ft(torch.tensor(vid).cuda()) # 1x101
    layerout = torch.tensor(layerout[0].numpy().transpose(1, 2, 3, 0)) #8x7x7x768
    pred_weights = model_ft.module.classifier.weight.data.detach().cpu().numpy().transpose() # 768 x 101
    pred = torch.argmax(predictions).item()
    cam = np.zeros(dtype = np.float32, shape = layerout.shape[0:3])
    for i, w in enumerate(pred_weights[:, label]):
    #i = 0, w:101
        # Compute cam for every kernel
        cam += w * layerout[:, :, :, i] # 8x7x7

    # Resize CAM to frame level
    cam = zoom(cam, (2, 32, 32))  # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)

    # normalize
    cam -= np.min(cam)
    cam /= np.max(cam) - np.min(cam)
    return cam, pred


def save_imgs(cam, pred, RGB_vid):
    # make dirs and filenames
    example_name = os.path.basename(frame_dir)
    heatmap_dir = os.path.join(base_output_dir, example_name, str(label), "heatmap")
    focusmap_dir = os.path.join(base_output_dir, example_name, str(label), "focusmap")
    for d in [heatmap_dir, focusmap_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    file = open(os.path.join(base_output_dir, example_name, str(label), "info.txt"), "a")
    file.write("Visualizing for class {}\n".format(label))
    file.write("Predicted class {}\n".format(pred))
    file.close()

    # produce heatmap and focusmap for every frame and activation map
    for i in range(0, cam.shape[0]):
        #   Create colourmap
        # COLORMAP_AUTUMN = 0,
        # COLORMAP_BONE = 1,
        # COLORMAP_JET = 2,
        # COLORMAP_WINTER = 3,
        # COLORMAP_RAINBOW = 4,
        # COLORMAP_OCEAN = 5,
        # COLORMAP_SUMMER = 6,
        # COLORMAP_SPRING = 7,
        # COLORMAP_COOL = 8,
        # COLORMAP_HSV = 9,
        # COLORMAP_PINK = 10,
        # COLORMAP_HOT = 11

        heatmap = cv2.applyColorMap(np.uint8(255 * cam[i]), cv2.COLORMAP_WINTER)
        #   Create focus map
        # focusmap = np.uint8(255 * cam[i])
        # focusmap = cv2.normalize(cam[i], dst=focusmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Create frame with heatmap
        heatframe = heatmap // 2 + RGB_vid[0][i] // 2
        cv2.imwrite(os.path.join(heatmap_dir, '{:03d}.png'.format(i)), heatframe)

        #   Create frame with focus map in the alpha channel
        focusframe = RGB_vid[0][i]
        focusframe = cv2.cvtColor(np.uint8(focusframe), cv2.COLOR_BGR2BGRA)
        focusframe[:, :, 3] = focusframe
        cv2.imwrite(os.path.join(focusmap_dir, '{:03d}.png'.format(i)), focusframe)


def feat_map_visualize(img_embedding, img_tensor, name='0'):
    # Resize CAM to frame level
    print(img_embedding.size())
    b, p, c = img_embedding.size()
    img_patch_embedding = img_embedding.permute(0, 2, 1).view(b, c, 14, 14)
    img_patch_embedding = F.interpolate(img_patch_embedding, size=(224, 224), mode='bilinear')
    print(img_patch_embedding.size())
    img_patch_embedding = torch.mean(img_patch_embedding, dim=1)
    cam = img_patch_embedding.cpu().detach().numpy()
    # normalize
    cam -= np.min(cam)
    cam /= np.max(cam) - np.min(cam)
    # print(img_patch_embedding.shape)
    # normalize
    # convert img_tensor to img
    img = img_tensor[0].permute(1, 2, 0).cpu().detach().numpy()
    # normalize
    img -= np.min(img)
    img /= np.max(img) - np.min(img)
    img = img * 255
    heatmap = cv2.applyColorMap(np.uint8(255 * cam[0]), cv2.COLORMAP_JET)
    heatframe = heatmap // 2 + img // 2
    vis_img = np.concatenate((heatframe, img), axis=1)
    cv2.imwrite('{}.png'.format(name), vis_img)
    print(name, "finished")
    return True


def cross_feat_map_visualize(img_embedding, text_embedding, caption, img_tensor, text_model, model_name='frozen', name='0'):
    # use text_embedding as query and measure the dot product
    words = caption.split(' ')
    start = 1
    # print(words)
    for i in range(len(words)):
        if model_name == 'clip':
            token_of_word = clip.tokenize(words[i])
            #print(words[i], torch.count_nonzero(token_of_word[0]))
            end = start + torch.count_nonzero(token_of_word[0]).item() - 2
        else:
            token_of_word = text_model.token_of_word(words[i])
            # print(words[i], token_of_word['input_ids'][0])
            end = start + len(token_of_word['input_ids'][0]) - 2
        if not check_nouns(words[i]):
            continue
        # print(start, end)
        # print(img_embedding.size())
        # print(text_embedding.size())
        # print(text_embedding[0, start:end, :].size())
        # Resize CAM to frame level
        # # ========= v0: directly apply dot ptoducts ========
        # img_embedding = img_embedding * torch.mean(text_embedding[0, start:end, :], dim=0)
        # b, p, c = img_embedding.size()
        # img_patch_embedding = img_embedding.permute(0, 2, 1).view(b, c, 14, 14)
        # img_patch_embedding = F.interpolate(img_patch_embedding, size=(224, 224), mode='bilinear')
        # img_patch_embedding = torch.mean(img_patch_embedding, dim=1)
        # ========= v1: text as query and measure attn weight

        # v1: get text embedding dirrectly
        if model_name == 'frozen':
            text_embedding = text_model(words[i])
        else:
            text_embedding = text_model(token_of_word.cuda())
        # norm
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)

        scale = 256 ** -0.5
        # v0: select part text embedding
        # q = torch.mean(text_embedding[0, start:end, :], dim=0).unsqueeze(0).unsqueeze(0)
        # v1: use word as input directly
        q = text_embedding[0][0].unsqueeze(0).unsqueeze(0)
        k = img_embedding
        # q = torch.relu(q)
        # k = torch.relu(k)
        # print(q.size(), k.size())
        dots = einsum('b i d, b j d -> b i j', q, k) * scale
        attn = dots.softmax(dim = -1)
        attn = attn.view(1, 1, 14, 14)
        img_patch_embedding = F.interpolate(attn, size=(224, 224), mode='bilinear').squeeze(0)
        # print(img_patch_embedding.size())

        cam = img_patch_embedding.cpu().detach().numpy()
        # normalize
        cam -= np.min(cam)
        cam /= np.max(cam) - np.min(cam)
        # print(img_tensor.size())
        # img = img_tensor.permute(1, 2, 0).cpu().detach().numpy()
        img = img_tensor[0].permute(1, 2, 0).cpu().detach().numpy()
        # normalize
        img -= np.min(img)
        img /= np.max(img) - np.min(img)
        img = img * 255
        # change color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam[0]), cv2.COLORMAP_JET)

        # if need?
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

        # add text
        position = (20, 20)
        scale_factor = 0.5
        params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
                  (0, 0, 255))
        cv2.putText(heatmap, words[i], *params)
        heatframe = heatmap // 2 + img // 2
        # add text
        # cv2.putText(heatframe, words[i], *params)
        vis_img = np.concatenate((heatframe, img), axis=1)
        # add caption
        text_shadow = np.ones([50, 224*2, 3]) * 255
        cap_len = len(caption)
        beg = 0
        j = 0
        max_char_per_line = 40
        while beg < cap_len:
            params = ((20, 20 * (j+1)), cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
                      (0, 0, 255))
            cv2.putText(text_shadow, caption[j*max_char_per_line:(j+1)*max_char_per_line], *params)
            j += 1
            beg += max_char_per_line
        cat_img = np.concatenate((vis_img, text_shadow), axis=0)
        cv2.imwrite('{}_token_{}.png'.format(name, i), cat_img)
        start = end
    print(name, "finished")
    return True


def cross_attention_visualize(img_embedding, img_tensor, caption, text_embedding, text_model, model_name='frozen', name='0', v=1):
    if v == 0:
        feat_map_visualize(img_embedding, img_tensor, name=name)
    elif v == 1:
        cross_feat_map_visualize(img_embedding, text_embedding, caption, img_tensor, text_model, model_name=model_name, name=name)
    else:
        print("not support now")
    return 0