import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid
from base.base_trainer import Multi_BaseTrainer_dist, BaseTrainer
from utils import inf_loop
from model.model import sim_matrix
from itertools import cycle
import sys
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
import time
from model.oa_loss import NCESoftmaxLoss, MemoryMoCo
torch.autograd.set_detect_anomaly(True)


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


# for distributed train
class Multi_Trainer_dist(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        # add cross modality contrastive loss for paired text and video embedding
        self.cm_contrast = MemoryMoCo(128, self.total_batch_sum * self.batch_size, 8092 * 16, 0.07, use_softmax=True).cuda()
        self.cm_criterion = NCESoftmaxLoss().cuda()
        # out = contrast(feat_q, feat_k, feat_n, index)
        # contrast_loss = criterion(out)


    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _pseudo_label_loss(self, predict, gt):
        loss_f = nn.BCELoss()
        return loss_f(predict, gt.type(torch.float))

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.learning_rate1
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)
        begin_time = time.time()
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                  truncation=True)
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                # print(data['text'])
                data['video'] = data['video'].to(self.device)
                if self.config['data_loader'][0]['args']['text_params']['object_tags']:
                    if self.tokenizer is not None:
                        data['pad_text'] = self.tokenizer(data['pad_text'], return_tensors='pt', padding=True,
                                                        truncation=True)
                    data['pad_text'] = {key: val.to(self.device) for key, val in data['pad_text'].items()}
                # print(data['text'][0], data['pad_text'][0])
                # for fine-grained match
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    text_embeds, pad_text_embeds, video_embeds, pad_video_embeds, \
                        [text_region_features, pad_text_region_features, video_region_feature,
                         object_region_feature, region_feat, tags_feat] \
                        = self.model(data)
                    # further cross modality attention model
                    # self.cross_model(text_embeds, video_embeds)
                    # out = self.cm_contrast(sim_matrix(text_embeds, video_embeds), sim_matrix(pad_text_embeds, video_embeds),
                    #                        sim_matrix(text_embeds.flip(0), video_embeds))

                    # gather data
                    video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                    pad_text_embeds = self.allgather(pad_text_embeds, self.n_gpu, self.args)
                    pad_video_embeds = self.allgather(pad_video_embeds, self.n_gpu, self.args)
                    text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)
                    #
                    # print(k.size())
                    # f_q = self.allgather(f_q, self.n_gpu, self.args)
                    # f_k = self.allgather(f_k, self.n_gpu, self.args)
                    # f_n = self.allgather(f_n, self.n_gpu, self.args)
                    #
                    region_feat = self.allgather(region_feat, self.n_gpu, self.args)
                    tags_feat = self.allgather(tags_feat, self.n_gpu, self.args)
                    # pad_text_region_features = self.allgather(pad_text_region_features, self.n_gpu, self.args)
                    # video_region_feature = self.allgather(video_region_feature, self.n_gpu, self.args)
                    # loss
                    # sim_matrix average from st2lv, st2sv?
                    output = sim_matrix(text_embeds, video_embeds)
                    st2sv_loss = self.loss(output)  # normal t2v loss
                    show_st2sv_loss = st2sv_loss.clone()
                    lt2sv_loss = self.loss(sim_matrix(pad_text_embeds, video_embeds))
                    # st2lv_loss = self.loss(sim_matrix(text_embeds, pad_video_embeds))
                    # out = self.cm_contrast(f_q, f_k, f_n)
                    # cm_contrast_loss = self.cm_criterion(out)
                    # lt2lv_loss = self.loss(sim_matrix(pad_text_embeds, pad_video_embeds))
                    # loss += lt2lv_loss
                    # loss = st2sv_loss + lt2sv_loss # + 1 * st2lv_loss # + 0.1 * cm_contrast_loss
                    loss = st2sv_loss + lt2sv_loss # + st2lv_loss
                    # loss += cm_contrast_loss
                    # == fine-grained loss 1
                    # for j in range(pad_text_region_features.size(1)):
                    #     loss += 0.1 * self.loss(
                    #         sim_matrix(pad_text_region_features[:, j, :], torch.mean(video_region_feature, dim=1)))
                    # # == word 2 object region match
                    # # include intra and inter
                    # fine_grained_loss = self.loss(sim_matrix(region_feat.view(-1, region_feat.size(2)), tags_feat.view(-1, tags_feat.size(2))))
                    # loss += fine_grained_loss
                    # == word 2 object region match
                    # only include inter sample
                    # fine_grained_loss = self.loss(sim_matrix(region_feat[:, 0, :], tags_feat[:, 0, :]))
                    fine_grained_loss = self.loss(sim_matrix(torch.mean(region_feat, dim=1), torch.mean(tags_feat, dim=1)))
                    loss += fine_grained_loss
                    # how to perform matrix comput here?
                    # print(patch_masks.size())
                    # print(object_token_lens.size())
                    # region_feat = video_region_feature * patch_masks
                    # tags_feat = text_region_features[object_token_lens:-1]
                    # word_patch_loss = self.loss(sim_matrix(region_feat, tags_feat))
                    # loss += word_patch_loss

                loss.backward()
                end_time = time.time()

                if batch_idx % self.log_step == 0:
                    if self.args.rank == 0:
                        # print("st2sv:{} lt2st:{}".format((loss - lt2st_loss).item(), lt2st_loss.item()))
                        # print("cm contra loss: {}".format(cm_contrast_loss))
                        print("time{} st2sv:{} lt2sv:{} fine_grained_loss:{}".format(end_time - begin_time,
                                                                                            show_st2sv_loss.item(),
                                                                                            lt2sv_loss.item(),
                                                                                            fine_grained_loss.item()))
                        # print("time{} st2sv:{} lt2sv:{} lv2st: {} fine_grained_loss:{}".format(end_time - begin_time,
                        #                                                                     show_st2sv_loss.item(),
                        #                                                                     lt2sv_loss.item(),
                        #                                                                     st2lv_loss.item(),
                        #                                                                     fine_grained_loss.item()))
                        # print("time{} st2sv:{} lt2sv:{} lv2st: {} cm_contra_loss:{}".format(end_time-begin_time, show_st2sv_loss.item(), lt2sv_loss.item(), st2lv_loss.item(), cm_contrast_loss.item()))
                        # print("time{} st2sv:{} lt2lv: {} cm_contra_loss:{}".format(end_time-begin_time, (loss-lt2lv_loss-cm_contrast_loss).item(), lt2lv_loss.item(), cm_contrast_loss.item()))
                        # print("time{} st2sv:{} lt2lv: {}".format(end_time - begin_time, (
                        #             loss - lt2lv_loss).item(), lt2lv_loss.item()))

                self.optimizer.step()
                if self.writer is not None and self.args.rank == 0:
                    self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    self.logger.debug('[{:.2f}s] Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        time.time()-begin_time,
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))
                    begin_time = time.time()
                self.optimizer.zero_grad()
            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        #if self.lr_scheduler is not None:
        #    self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        object_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        pad_vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        if self.config['arch']['args']['text_params']['two_outputs']:
            pad_text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    # print(data['text']) # input_ids and input_masks
                    data['video'] = data['video'].to(self.device)
                    if self.config['data_loader'][0]['args']['text_params']['object_tags']:
                        if self.tokenizer is not None:
                            data['pad_text'] = self.tokenizer(data['pad_text'], return_tensors='pt', padding=True,
                                                              truncation=True)
                        data['pad_text'] = {key: val.to(self.device) for key, val in data['pad_text'].items()}

                    # for fine-grained match
                    data['patch_masks'] = data['patch_masks'].to(self.device)
                    data['object_token_len'] = data['object_token_len'].to(self.device)
                    data['object_token_masks'] = data['object_token_masks'].to(self.device)
                    data['object_token_len'] = data['object_token_len'].to(self.device)
                    # print(patch_masks.size(), object_token_lens.size())
                    # print(data['text']['input_ids'][0], data['pad_text']['input_ids'][0])
                    text_embed, pad_text_embed, vid_embed, pad_vid_embed, \
                        [text_region_features, pad_text_region_features, video_region_feature, object_region_feature,
                         region_feat, tags_feat] \
                        = self.model.module(
                        data, return_embeds=True)
                    # print(q.size(), k.size(), n.size())
                    if vid_embed is not None:
                        vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(vid_embed_all, vid_embed)
                        vid_embed_all = torch.cat(vid_embed_all, dim=0)
                    if pad_vid_embed is not None:
                        pad_vid_embed_all = [torch.zeros_like(pad_vid_embed) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(pad_vid_embed_all, pad_vid_embed)
                        pad_vid_embed_all = torch.cat(pad_vid_embed_all, dim=0)
                    if text_embed is not None:
                        text_embed_all = [torch.zeros_like(text_embed) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(text_embed_all, text_embed)
                        text_embed_all = torch.cat(text_embed_all, dim=0)
                    if pad_text_embed is not None:
                        pad_text_embed_all = [torch.zeros_like(pad_text_embed) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(pad_text_embed_all, pad_text_embed)
                        pad_text_embed_all = torch.cat(pad_text_embed_all, dim=0)
                    # add for fine-grained loss
                    # print(object_region_feature.size())
                    # if object_region_feature is not None:
                    #     object_region_feature_all = [torch.zeros_like(object_region_feature) for _ in range(self.n_gpu)]
                    #     torch.distributed.all_gather(object_region_feature_all, object_region_feature)
                    #     object_region_feature_all = torch.cat(object_region_feature_all, dim=0)

                    # =====
                    text_embed_arr[dl_idx].append(text_embed_all.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed_all.cpu())
                    pad_text_embed_arr[dl_idx].append(pad_text_embed_all.cpu())
                    pad_vid_embed_arr[dl_idx].append(pad_vid_embed_all.cpu())
                    sims_batch = sim_matrix(text_embed_all, vid_embed_all)
                    st2sv_loss = self.loss(sims_batch)  # normal video to text loss
                    loss = st2sv_loss
                    lt2sv_loss = self.loss(sim_matrix(pad_text_embed_all, vid_embed_all))
                    loss += lt2sv_loss


                    # lt2st_loss = self.loss(sim_matrix(pad_text_embed_all, text_embed_all))
                    # loss += lt2st_loss
                    if batch_idx % self.log_step == 0:
                        if self.args.rank == 0:
                            print("st2sv:{} lt2sv:{}".format((loss - lt2sv_loss).item(), lt2sv_loss.item()))
                    total_val_loss[dl_idx] += loss.item()

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}
            if self.config['arch']['args']['video_params']['two_outputs']:
                nested_st_to_lv_metrics = {x: {} for x in range(len(self.valid_data_loader))}
                if self.config['arch']['args']['text_params']['two_outputs']:
                    nested_lt_to_lv_metrics = {x: {} for x in range(len(self.valid_data_loader))}
            else:
                if self.config['arch']['args']['text_params']['two_outputs']:
                    if self.config['arch']['args']['video_params']['model'] == "" and \
                            self.config['arch']['args']['object_params']['model'] != "":
                        nested_lt_to_o_metrics = {x: {} for x in range(len(self.valid_data_loader))}
                    else:
                        nested_lt_to_sv_metrics = {x: {} for x in range(len(self.valid_data_loader))}
                        nested_lt_to_lv_metrics = {x: {} for x in range(len(self.valid_data_loader))}
            text_embeds = torch.cat(text_embed_arr[dl_idx])
            if self.config['arch']['args']['video_params']['model'] == "" and \
                    self.config['arch']['args']['object_params']['model'] != "":
                object_embeds = torch.cat(object_embed_arr[dl_idx])
                o2t_sims = sim_matrix(text_embeds, object_embeds).detach().cpu().numpy()
            else:
                vid_embeds = torch.cat(vid_embed_arr[dl_idx])
                st2sv_sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()
                # print(st2sv_sims.size())
            if self.config['arch']['args']['video_params']['two_outputs']:
                pad_vid_embeds = torch.cat(pad_vid_embed_arr[dl_idx])
                if self.config['arch']['args']['text_params']['two_outputs']:
                    pad_text_embeds = torch.cat(pad_text_embed_arr[dl_idx])
                    lt2lv_sims = sim_matrix(pad_text_embeds, pad_vid_embeds).detach().cpu().numpy()
                else:
                    st2lv_sims = sim_matrix(text_embeds, pad_vid_embeds).detach().cpu().numpy()
            else:
                if self.config['arch']['args']['text_params']['two_outputs']:
                    pad_text_embeds = torch.cat(pad_text_embed_arr[dl_idx])
                    pad_vid_embeds = torch.cat(pad_vid_embed_arr[dl_idx])
                    if self.config['arch']['args']['video_params']['model'] == "" and \
                            self.config['arch']['args']['object_params']['model'] != "":
                        lt2o_sims = sim_matrix(pad_text_embeds, text_embeds).detach().cpu().numpy()
                    else:
                        lt2sv_sims = sim_matrix(pad_text_embeds, vid_embeds).detach().cpu().numpy()
                        lt2lv_sims = sim_matrix(pad_text_embeds, pad_vid_embeds).detach().cpu().numpy()
            for metric in self.metrics:
                metric_name = metric.__name__
                if self.config['arch']['args']['video_params']['model'] == "" and \
                        self.config['arch']['args']['object_params']['model'] != "":
                    res = metric(o2t_sims)
                else:
                    res = metric(st2sv_sims)
                if self.args.rank == 0:
                    print("short text to short video")
                if self.args.rank == 0:
                    verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res
                if self.writer is not None:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

                if self.visualizer is not None:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(st2sv_sims, epoch, meta_arr_cat, nested_metrics)
            if self.config['arch']['args']['video_params']['two_outputs']:
                if self.config['arch']['args']['text_params']['two_outputs']:
                    if self.args.rank == 0:
                        print("long text to long video:")
                    for metric in self.metrics:
                        metric_name = metric.__name__
                        lt2lv_pad_res = metric(lt2lv_sims)
                        if self.args.rank == 0:
                            verbose(epoch=epoch, metrics=lt2lv_pad_res, name=self.valid_data_loader[dl_idx].dataset_name,
                                    mode=metric_name)
                        nested_lt_to_lv_metrics[dl_idx][metric_name] = lt2lv_pad_res
                else:
                    if self.args.rank == 0:
                        print("long video to short text:")
                    for metric in self.metrics:
                        metric_name = metric.__name__
                        st2lv_pad_res = metric(st2lv_sims)
                        if self.args.rank == 0:
                            verbose(epoch=epoch, metrics=st2lv_pad_res, name=self.valid_data_loader[dl_idx].dataset_name,
                                    mode=metric_name)
                        nested_lt_to_lv_metrics[dl_idx][metric_name] = st2lv_pad_res
            else:
                if self.config['arch']['args']['text_params']['two_outputs']:
                    if self.config['arch']['args']['video_params']['model'] == "" and \
                            self.config['arch']['args']['object_params']['model'] != "":
                        if self.args.rank == 0:
                            print("long text to object")
                        for metric in self.metrics:
                            metric_name = metric.__name__
                            lt2o_pad_res = metric(lt2o_sims)
                            if self.args.rank == 0:
                                verbose(epoch=epoch, metrics=lt2o_pad_res,
                                        name=self.valid_data_loader[dl_idx].dataset_name,
                                        mode=metric_name)
                            nested_lt_to_o_metrics[dl_idx][metric_name] = lt2o_pad_res
                    else:
                        # if self.args.rank == 0:
                        #     print("long text to short video")
                        # for metric in self.metrics:
                        #     metric_name = metric.__name__
                        #     lt2sv_pad_res = metric(lt2sv_sims)
                        #     if self.args.rank == 0:
                        #         verbose(epoch=epoch, metrics=lt2sv_pad_res,
                        #                 name=self.valid_data_loader[dl_idx].dataset_name,
                        #                 mode=metric_name)
                        #     nested_lt_to_sv_metrics[dl_idx][metric_name] = lt2sv_pad_res
                        if self.args.rank == 0:
                            print("long text to long video")
                        for metric in self.metrics:
                            metric_name = metric.__name__
                            lt2lv_pad_res = metric(lt2lv_sims)
                            if self.args.rank == 0:
                                verbose(epoch=epoch, metrics=lt2lv_pad_res,
                                        name=self.valid_data_loader[dl_idx].dataset_name,
                                        mode=metric_name)
                            nested_lt_to_lv_metrics[dl_idx][metric_name] = lt2lv_pad_res
        res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    for dl_idx in range(len(self.valid_data_loader))}
        res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples/self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            # take the min
            self.len_epoch = min([len(x) for x in data_loader])
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data_li in enumerate(zip(*self.data_loader)):
            if (batch_idx + 1) * self.total_batch_sum > self.max_samples_per_epoch:
                break
            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True,
                                                  truncation=True)
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                data['video'] = data['video'].to(self.device)

                self.optimizer.zero_grad()
                text_embeds, video_embeds = self.model(data)
                output = sim_matrix(text_embeds, video_embeds)
                loss = self.loss(output)
                loss.backward()
                #for name, param in self.model.named_parameters():
                #    if 'video_model.blocks.0.norm1.weight' in name:
                #        print(batch_idx, name, param.grad.cpu().numpy()[0:10])
                self.optimizer.step()
                if self.writer is not None:
                    self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} dl{} {} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss.detach().item()))
                    #for name, param in self.model.named_parameters():
                    #    if 'video_model.temporal_embed' in name:
                    #        print(param[:,:,0])
                self.optimizer.zero_grad()

            if batch_idx == self.len_epoch:
                break

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}
                    data['video'] = data['video'].to(self.device)

                    if isinstance(self.model, nn.DataParallel) and data["video"].shape[0] < len(self.model.device_ids):
                        # Note that if some batch has size smaller than the GPU size, `DataParallel` will fail.
                        # It can happen with the last batch of the dataset, depending on its size.
                        # This avoids using `DataParallel` in this case, and supposes the entire batch fits in one GPU.
                        text_embed, vid_embed = self.model.module(data, return_embeds=True)
                    else:
                        text_embed, vid_embed = self.model(data, return_embeds=True)

                    text_embed_arr[dl_idx].append(text_embed.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed.cpu())
                    sims_batch = sim_matrix(text_embed, vid_embed)
                    loss = self.loss(sims_batch)
                    total_val_loss[dl_idx] += loss.item()

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            text_embeds = torch.cat(text_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])
            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims)
                verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                        mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

                if self.visualizer is not None:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx])
                    for dl_idx in range(len(self.valid_data_loader))}
        res_dict['nested_val_metrics'] = nested_metrics

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
