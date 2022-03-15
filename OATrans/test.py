import argparse
import collections
import os
import torch
from tqdm import tqdm
from OATrans.data_loader import data_loader as module_data
from OATrans import model as module_metric, model as module_arch
from parse_config_dist_multi import ConfigParser
from OATrans.model.model import sim_matrix
import pandas as pd
from sacred import Experiment
import transformers
from utils.util import state_dict_data_parallel_fix
import torch.distributed as dist
from trainer.trainer_dist import verbose
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
    config._config['data_loader']['args']['split'] = 'test'
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
    if config['arch']['stream'] == 3:
        object_embed_arr = []
    if config['arch']['args']['video_params']['model'] == "" and \
            config['arch']['args']['object_params']['model'] != "":
        object_embed_arr = []
    if config['arch']['stream'] == 4:
        pad_text_embed_arr = []
        pad_vid_embed_arr = []
    if config['arch']['args']['text_params']['two_outputs']:
        pad_text_embed_arr = []
    print(len(data_loader))
    with torch.no_grad():
        for i, data in tqdm(tqdm(enumerate(data_loader))):
            # leave this for now since not doing anything on the gpu
            meta_arr.append(data['meta'])
            # print("caption is: {}".format(data['text'][0]))
            # print("object tags is: {}".format(data['pad_text'][0]))
            if tokenizer is not None:
                data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}
            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)
            if config['data_loader']['args']['text_params']['object_tags']:
                if tokenizer is not None:
                    data['pad_text'] = tokenizer(data['pad_text'], return_tensors='pt', padding=True,
                                                    truncation=True)
                data['pad_text'] = {key: val.to(device) for key, val in data['pad_text'].items()}
            # if config['arch']['object'] is True:
            #     data['object'] = data['object'].cuda()
            if config['data_loader']['args']['object_params']['input_objects'] is True:
                # pdb.set_trace()
                data['object'] = data['object'].cuda()
            text_embed, pad_text_embed, vid_embed, pad_vid_embed, object_embed, _ = model(
                data, return_embeds=True)
            if torch.is_tensor(pad_text_embed):
                pad_text_embed = allgather(pad_text_embed, config['n_gpu'], args)
            if torch.is_tensor(text_embed):
                text_embed = allgather(text_embed, config['n_gpu'], args)
            if torch.is_tensor(vid_embed):
                vid_embed = allgather(vid_embed, config['n_gpu'], args)
            if torch.is_tensor(pad_vid_embed):
                pad_vid_embed = allgather(pad_vid_embed, config['n_gpu'], args)
            if torch.is_tensor(object_embed):
                object_embed = allgather(object_embed, config['n_gpu'], args)
            # vid_embed = vid_embed/2 + pad_vid_embed/2
            text_embed_arr.append(text_embed)
            if config['arch']['args']['video_params']['model'] == "" and \
                    config['arch']['args']['object_params']['model'] != "":
                object_embed_arr.append(object_embed)
            else:
                vid_embed_arr.append(vid_embed)
            if config['arch']['stream'] == 3:
                object_embed_arr.append(object_embed)
            if config['arch']['args']['video_params']['two_outputs']:
                pad_vid_embed_arr.append(pad_vid_embed)
            if config['arch']['args']['text_params']['two_outputs']:
                pad_text_embed_arr.append(pad_text_embed)
    text_embeds = torch.cat(text_embed_arr)
    if config['arch']['args']['video_params']['model'] == "" and \
            config['arch']['args']['object_params']['model'] != "":
        object_embeds = torch.cat(object_embed_arr)
    else:
        vid_embeds = torch.cat(vid_embed_arr)
    if config['arch']['stream'] == 3:
        object_embeds = torch.cat(object_embed_arr)
    if config['arch']['args']['video_params']['two_outputs']:
        pad_vid_embeds = torch.cat(pad_vid_embed_arr)
    if config['arch']['args']['text_params']['two_outputs']:
        pad_text_embeds = torch.cat(pad_text_embed_arr)
    mask = None
    if data_loader.dataset.sliding_window_stride != -1:
        if config['arch']['args']['video_params']['model'] == "" and \
                config['arch']['args']['object_params']['model'] != "":
            cpu_object_embeds = object_embeds.cpu().detach()
        else:
            cpu_vid_embeds = vid_embeds.cpu().detach()
        cpu_text_embeds = text_embeds.cpu().detach()
        if config['arch']['stream'] == 3:
            cpu_object_embeds = object_embeds.cpu().detach()
        if config['arch']['args']['video_params']['two_outputs']:
            cpu_pad_vid_embeds = pad_vid_embeds.cpu().detach()
        if config['arch']['args']['text_params']['two_outputs']:
            cpu_pad_text_embeds = pad_text_embeds.cpu().detach()
        if config['arch']['args']['video_params']['model'] == "" and \
                config['arch']['args']['object_params']['model'] != "":
            li_object_embeds = [x for x in cpu_object_embeds]
        else:
            li_vid_embeds = [x for x in cpu_vid_embeds]
        li_txt_embeds = [x for x in cpu_text_embeds]
        if config['arch']['stream'] == 3:
            li_object_embeds = [x for x in cpu_object_embeds]
        if config['arch']['args']['video_params']['two_outputs']:
            li_pad_vid_embeds = [x for x in cpu_pad_vid_embeds]
        if config['arch']['args']['text_params']['two_outputs']:
            li_pad_txt_embeds = [x for x in cpu_pad_text_embeds]
        videoids = pd.Series([x['paths'] for x in meta_arr]).explode()
        raw_caps = pd.Series([x['raw_captions']] for x in meta_arr).explode().explode()
        if config['arch']['stream'] == 3:
            vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                                   'object_embed': li_object_embeds,  'captions': raw_caps})
        if config['arch']['args']['video_params']['model'] == "" and \
                config['arch']['args']['object_params']['model'] != "":
            if config['arch']['args']['text_params']['two_outputs']:
                vid_df = pd.DataFrame({'videoid': videoids, 'txt_embed': li_txt_embeds,
                                       'object_embed': li_object_embeds, 'pad_text_embed': li_pad_txt_embeds, 'captions': raw_caps})
            else:
                vid_df = pd.DataFrame({'videoid': videoids, 'txt_embed': li_txt_embeds,
                                       'object_embed': li_object_embeds,
                                       'captions': raw_caps})
        else:
            if config['arch']['args']['video_params']['two_outputs']:
                if config['arch']['args']['text_params']['two_outputs']:
                    vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                                           'pad_vid_embed': li_pad_vid_embeds, 'pad_text_embed': li_pad_txt_embeds,
                                           'object_embed': li_object_embeds, 'captions': raw_caps})
                else:
                    vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                                           'pad_vid_embed': li_pad_vid_embeds,
                                           'object_embed': li_object_embeds, 'captions': raw_caps})
            else:
                if config['arch']['args']['text_params']['two_outputs']:
                    vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                                            'pad_text_embed': li_pad_txt_embeds, 'captions': raw_caps})
                else:
                    vid_df = pd.DataFrame({'videoid': videoids, 'vid_embed': li_vid_embeds, 'txt_embed': li_txt_embeds,
                                           'captions': raw_caps})
        new_vid_embeds = []
        new_txt_embeds = []
        new_object_embeds = []
        new_pad_vid_embeds = []
        new_pad_txt_embeds = []
        for vid in vid_df['videoid'].unique():
            tdf = vid_df[vid_df['videoid'] == vid]
            tvembeds = torch.stack(tdf['vid_embed'].values.tolist())
            tvembeds = tvembeds.mean(dim=0)
            new_vid_embeds.append(tvembeds)
            if config['arch']['args']['video_params']['model'] == "" and \
                    config['arch']['args']['object_params']['model'] != "":
                toembeds = torch.stack(tdf['object_embed'].values.tolist())
                toembeds = toembeds.mean(dim=0)
                new_object_embeds.append(toembeds)

            for cap in tdf['captions'].unique():
                cdf = vid_df[vid_df['captions'] == cap]
                ttembeds = torch.stack(cdf['txt_embed'].values.tolist())
                new_txt_embeds.append(ttembeds[0])
            if config['arch']['args']['video_params']['two_outputs']:
                pad_tvembeds = torch.stack(tdf['pad_vid_embed'].values.tolist())
                pad_tvembeds = pad_tvembeds.mean(dim=0)
                new_pad_vid_embeds.append(pad_tvembeds)
            if config['arch']['args']['text_params']['two_outputs']:
                for cap in tdf['captions'].unique():
                    cdf = vid_df[vid_df['captions'] == cap]
                    pad_ttembeds = torch.stack(cdf['pad_txt_embed'].values.tolist())
                    new_pad_txt_embeds.append(pad_ttembeds[0])

        if config['arch']['args']['video_params']['model'] == "" and \
                config['arch']['args']['object_params']['model'] != "":
            object_embeds = torch.stack(new_object_embeds).cuda()
        else:
            vid_embeds = torch.stack(new_vid_embeds).cuda()
        text_embeds = torch.stack(new_txt_embeds).cuda()
        if config['arch']['stream'] == 3:
            object_embeds = torch.stack(new_object_embeds).cuda()
        if config['arch']['args']['video_params']['two_outputs']:
            pad_vid_embeds = torch.stack(new_pad_vid_embeds).cuda()
        if config['arch']['args']['text_params']['two_outputs']:
            pad_text_embeds = torch.stack(new_pad_txt_embeds).cuda()
    if config['arch']['args']['video_params']['model'] == "" and \
            config['arch']['args']['object_params']['model'] != "":
        sims = sim_matrix(text_embeds, object_embeds)
    else:
        sims = sim_matrix(text_embeds, vid_embeds)
    sims = sims.detach().cpu().numpy()
    if config['arch']['stream'] == 3:
        sims_o2v = sim_matrix(object_embeds, vid_embeds)
        sims_o2v = sims_o2v.detach().cpu().numpy()
        sims_o2t = sim_matrix(text_embeds, object_embeds)
        sims_o2t = sims_o2t.detach().cpu().numpy()
    if config['arch']['args']['video_params']['two_outputs']:
        sims_st2lv = sim_matrix(text_embeds, pad_vid_embeds)
        sims_st2lv = sims_st2lv.detach().cpu().numpy()
        if config['arch']['args']['text_params']['two_outputs']:
            sims_lt2sv = sim_matrix(pad_text_embeds, vid_embeds)
            sims_lt2sv = sims_lt2sv.detach().cpu().numpy()
            sims_lt2lv = sim_matrix(pad_text_embeds, pad_vid_embeds)
            sims_lt2lv = sims_lt2lv.detach().cpu().numpy()
    else:
        if config['arch']['args']['text_params']['two_outputs']:
            if config['arch']['args']['video_params']['model'] == "" and \
                    config['arch']['args']['object_params']['model'] != "":
                sims_lt2o = sim_matrix(pad_text_embeds, object_embeds)
                sims_lt2o = sims_lt2o.detach().cpu().numpy()
            else:
                sims_lt2sv = sim_matrix(pad_text_embeds, vid_embeds)
                sims_lt2sv = sims_lt2sv.detach().cpu().numpy()
                sims_lt2st = sim_matrix(pad_text_embeds, text_embeds)
                sims_lt2st = sims_lt2st.detach().cpu().numpy()
    if args.rank == 0:
        print("normal text to normal video: ")
        nested_metrics = {}
        for metric in metric_fns:
            metric_name = metric.__name__
            res = metric(sims, query_masks=mask)
            verbose(epoch=0, metrics=res, name="", mode=metric_name)
            nested_metrics[metric_name] = res
        if config['arch']['stream'] == 3:
            print("object to video: ")
            o2v_nested_metrics = {}
            for metric in metric_fns:
                metric_name = metric.__name__
                res = metric(sims_o2v, query_masks=mask)
                verbose(epoch=0, metrics=res, name="", mode=metric_name)
                o2v_nested_metrics[metric_name] = res
            print("text to object: ")
            o2t_nested_metrics = {}
            for metric in metric_fns:
                metric_name = metric.__name__
                res = metric(sims_o2t, query_masks=mask)
                verbose(epoch=0, metrics=res, name="", mode=metric_name)
                o2t_nested_metrics[metric_name] = res
        if config['arch']['args']['video_params']['two_outputs']:
            print("normal text to long video: ")
            st2lv_nested_metrics = {}
            for metric in metric_fns:
                metric_name = metric.__name__
                res = metric(sims_st2lv, query_masks=mask)
                verbose(epoch=0, metrics=res, name="", mode=metric_name)
                st2lv_nested_metrics[metric_name] = res
            if config['arch']['args']['text_params']['two_outputs']:
                print("long video to long text: ")
                lt2lv_nested_metrics = {}
                for metric in metric_fns:
                    metric_name = metric.__name__
                    res = metric(sims_lt2lv, query_masks=mask)
                    verbose(epoch=0, metrics=res, name="", mode=metric_name)
                    lt2lv_nested_metrics[metric_name] = res
        else:
            if config['arch']['args']['text_params']['two_outputs']:
                if config['arch']['args']['video_params']['model'] == "" and \
                        config['arch']['args']['object_params']['model'] != "":
                    print("long text to normal object: ")
                    lt2o_nested_metrics = {}
                    for metric in metric_fns:
                        metric_name = metric.__name__
                        res = metric(sims_lt2o, query_masks=mask)
                        verbose(epoch=0, metrics=res, name="", mode=metric_name)
                        lt2o_nested_metrics[metric_name] = res
                else:
                    print("long text to normal video: ")
                    lt2sv_nested_metrics = {}
                    for metric in metric_fns:
                        metric_name = metric.__name__
                        res = metric(sims_lt2sv, query_masks=mask)
                        verbose(epoch=0, metrics=res, name="", mode=metric_name)
                        lt2sv_nested_metrics[metric_name] = res
        print("long text to short text:")
        lt2st_nested_metrics = {}
        for metric in metric_fns:
            metric_name = metric.__name__
            res = metric(sims_lt2st, query_masks=mask)
            verbose(epoch=0, metrics=res, name="", mode=metric_name)
            lt2st_nested_metrics[metric_name] = res


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
