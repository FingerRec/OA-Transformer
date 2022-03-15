import argparse
import collections
import os
import torch
from tqdm import tqdm
import data_loader.data_loader as module_data
import model.metric as module_metric
import model.oa_model_region_mem as module_arch
from parse_config_dist_multi import ConfigParser
from model.model import sim_matrix
import pandas as pd
import numpy as np
from sacred import Experiment
import transformers
from utils.util import state_dict_data_parallel_fix
import pdb
import torch.distributed as dist
from trainer.trainer_dist import verbose
import torch.nn.functional as F
import cv2
ex = Experiment('test')


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(n_gpu)]
        dist.all_gather(output, tensor)
        ctx.local_rank = args.local_rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.local_rank : ctx.batch_size * (ctx.local_rank + 1)],
            None, None,
        )


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


@ex.main
def run():
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://{}:{}'.format(
                                             args.master_address, args.master_port),
                                         rank=args.rank, world_size=args.world_size)

    # setup data_loader instances
    config._config['data_loader']['args']['split'] = 'val'
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['sliding_window_stride'] = config._config['sliding_window_stride']
    data_loader = config.initialize('data_loader', module_data)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    #logger.info('Loading checkpoint: {} ...'.format(config.resume))

    # pdb.set_trace()
    if config['arch']['args']['load_checkpoint'] is not None:
        checkpoint = torch.load(config['arch']['args']['load_checkpoint'])
        state_dict = checkpoint['state_dict']
        # print(state_dict)
        # for k, v in state_dict.items():
        #     if 'video_model' in k:
        #         print(k)
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        # model.load_state_dict(new_state_dict, strict=True)
        model.load_state_dict(new_state_dict, strict=False)
        print("load pretrained model")
        # model.load_state_dict(state_dict, strict=False)
    else:
        print('Using random weights')

    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # gather data
    allgather = AllGather.apply

    meta_arr = []
    text_embed_arr = []
    vid_embed_arr = []
    print(len(data_loader))
    with torch.no_grad():
        for i, data in tqdm(tqdm(enumerate(data_loader))):
            # leave this for now since not doing anything on the gpu
            meta_arr.append(data['meta'])
            if tokenizer is not None:
                data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}
            data['video'] = data['video'].to(device)
            data['text_region_embedding'] = data['text_region_embedding'].to(device)
            data['patch_masks'] = data['patch_masks'].to(device)
            text_embed, vid_embed, region_sim = model(data)

            # visualize here
            # save prediction and attention map (prediction result)
            # region_sim[0][0].cpu().numpy() # resize to 224 x 224
            region_sim_map = region_sim[0][0].view(14, 14).unsqueeze(0).unsqueeze(0)*255
            region_sim_map = F.interpolate(region_sim_map, size=(224, 224), mode='bilinear')
            region_sim_map_gray = region_sim_map[0][0].cpu().detach().numpy()
            region_sim_map_rgb = cv2.cvtColor(region_sim_map_gray, cv2.COLOR_GRAY2RGB)

            # data['patch_masks'][0][0].cpu().numpy() # resize to 224 x 224
            gt_masks_map = data['patch_masks'][0][0].view(14, 14).unsqueeze(0).unsqueeze(0)*255
            gt_masks_map = F.interpolate(gt_masks_map, size=(224, 224), mode='bilinear')
            # print(gt_masks_map.size())
            gt_masks_map_gray = gt_masks_map[0][0].cpu().detach().numpy()
            gt_masks_map_rgb = cv2.cvtColor(gt_masks_map_gray.astype('uint8'), cv2.COLOR_GRAY2RGB)

            #print(text_region_map_rgb.shape)
            # original image data['meta']['paths'][0] resize to 224 x 224

            # visualize objects

            data_form = 'video' # 'image'
            #print(data['meta']['paths'][0])
            if data_form == 'video':
                # print(data['meta']['paths'][0])
                cap = cv2.VideoCapture(data['meta']['paths'][0])
                assert (cap.isOpened())
                print(data['meta']['idxs'][0])
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(data['meta']['idxs'][0][0]) - 1)
                ret, orig_img = cap.read()
            else:
                orig_img = cv2.imread(data['meta']['paths'][0])
            resized = cv2.resize(orig_img, (224, 224), interpolation=cv2.INTER_AREA)
            object_text = data['meta']['top_1_object'][0]
            # add text
            position = (20, 20)
            scale_factor = 0.5
            params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
                      (0, 0, 255))
            cv2.putText(resized, object_text, *params)

            #
            concated_img = np.concatenate((resized, gt_masks_map_rgb, region_sim_map_rgb), axis=1)
            cv2.imwrite('utils/visualization/transfer_predict_visualization/{}_predict.png'.format(i), concated_img)
            # generate object list data['meta']['top_1_object']

            if torch.is_tensor(text_embed):
                text_embed = allgather(text_embed, config['n_gpu'], args)
            if torch.is_tensor(vid_embed):
                vid_embed = allgather(vid_embed, config['n_gpu'], args)
            if torch.is_tensor(region_sim):
                region_sim = allgather(region_sim, config['n_gpu'], args)
            text_embed_arr.append(text_embed)
            vid_embed_arr.append(vid_embed)
    text_embeds = torch.cat(text_embed_arr)
    vid_embeds = torch.cat(vid_embed_arr)
    mask = None
    if data_loader.dataset.sliding_window_stride != -1:
        cpu_vid_embeds = vid_embeds.cpu().detach()
        cpu_text_embeds = text_embeds.cpu().detach()
        li_vid_embeds = [x for x in cpu_vid_embeds]
        li_txt_embeds = [x for x in cpu_text_embeds]
        videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
        raw_caps = pd.Series([x['raw_captions']] for x in meta_arr).explode().explode()
        vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                               'captions': raw_caps})
        new_vid_embeds = []
        new_txt_embeds = []
        for vid in vid_df['videoid'].unique():
            tdf = vid_df[vid_df['videoid'] == vid]
            tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
            tvembeds = tvembeds.mean(dim=0)
            new_vid_embeds.append(tvembeds)
            for cap in tdf['captions'].unique():
                cdf = vid_df[vid_df['captions'] == cap]
                ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                new_txt_embeds.append(ttembeds[0])

        vid_embeds = torch.stack(new_vid_embeds).cuda()
        text_embeds = torch.stack(new_txt_embeds).cuda()
    sims = sim_matrix(text_embeds, vid_embeds)
    sims = sims.detach().cpu().numpy()
    if args.rank == 0:
        print("normal text to normal video: ")
        nested_metrics = {}
        for metric in metric_fns:
            metric_name = metric.__name__
            res = metric(sims, query_masks=mask)
            verbose(epoch=0, metrics=res, name="", mode=metric_name)
            nested_metrics[metric_name] = res


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    args.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
    args.add_argument('-k', '--local_rank', type=int, default=0)
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')

    master_address = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])
    world_size = int(os.environ['WORLD_SIZE'])
    # world_size = int(torch.cuda.device_count())
    rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    args.add_argument('-ma', '--master_address', default=master_address)
    args.add_argument('-mp', '--master_port', type=int, default=master_port)
    args.add_argument('-ws', '--world_size', type=int, default=world_size)
    args.add_argument('-rk', '--rank', type=int, default=rank)
    args.add_argument('-lr1', '--learning_rate1', type=float, default=2e-4)
    args.add_argument('-sc', '--schedule', default=[60, 80])

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]
    config = ConfigParser(args, test=True)
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    ex.run()
