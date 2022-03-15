import torch.nn as nn
import torch as th
import torch.nn.functional as F
import torch
import math
import numpy as np
from model.model import sim_matrix
from model.loss import NormSoftmaxLoss
from torch.autograd import Variable



# simsiam loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    return F.mse_loss(input_logits, target_logits, size_average=False)  # / num_classes
    # input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    # num_classes = input_logits.size()[1]
    # return F.mse_loss(input_softmax, target_softmax, size_average=False)  # / num_classes


# n_data = len(dataset)
# contrast = MemoryMoCo(128, n_data, 8092*4, 0.07, use_softmax=True).cuda()
# criterion = NCESoftmaxLoss()
# criterion = criterion.cuda()
#
# out = contrast(feat_q, feat_k, feat_n, index)
# contrast_loss = criterion(out)


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss

class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    # T = 0.2 achieve best result?
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        # self.register_buffer('spatial_memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k, n):
        # n, sn,
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # # neg logit
        # # queue = self.memory_bank.get_queue(self.queueSize, indexs)
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)
        #out = torch.cat((l_pos, l_neg), dim=1)

        # other negative
        l_neg_2 = torch.bmm(q.view(batchSize, 1, -1), n.view(batchSize, -1, 1))
        l_neg_2 = l_neg_2.view(batchSize, 1)
        #
        # strong negative
        # l_s_neg = torch.bmm(q.view(batchSize, 1, -1), sn.view(batchSize, -1, 1))
        # l_s_neg = l_s_neg.view(batchSize, 1)

        out = torch.cat((l_pos, l_neg, l_neg_2), dim=1)
        # out = torch.cat((l_pos, l_neg, l_neg_2, l_s_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # label = torch.zeros([batchSize]).cuda().long()
        # loss = []
        # for i in range(batchSize):
        #     loss.append(self.criterion(out[i].unsqueeze(0), label[i].unsqueeze(0)))
        # print(loss)
        # self.memory_bank.batch_set(indexs, k, loss)
        # self.memory = self.memory_bank.update_queue(self.memory)
        # print(self.memory_bank.link)
        # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize) # 1 fmod 1.5 = 1  2 fmod 1.5 = 0.5
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize
        # add for spatial memory

        return out


class FineGrainedLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.criterion = NormSoftmaxLoss(temperature)

    def forward(self, vid_feats, text_feats, bboxs, object_token_len, real_len):
        # find the patch that contain in bboxes
        loss = None
        bboxs[:, :4] = bboxs[:, :4] * 16
        bboxs[:, :2] = torch.round(bboxs[:, :2])
        bboxs[:, 2:4] = torch.ceil(bboxs[:, 2:4])
        # for each sample
        # print(vid_feats.size(), text_feats.size()) # 128 x 196 x 256, 128 x 14 x 256

        # step1: for each bbox, get corresponding features in tensor [B, 10, 256]
        for index, bbox in enumerate(bboxs):
            patch_indexs = np.zeros(16*16)
            for i in range(16):
                for j in range(16):
                    if i > bbox[:, 0] and i < bbox[:, 2] and j > bbox[:, 1] and j < bbox[:, 3]:
                        patch_indexs[:, i*16+j] = 1
            # select patch features according to indexs
            vid_feats_related = vid_feats[:, patch_indexs]
            vid_feat = torch.mean(vid_feats_related, dim=1)
            # shared proj head ?

        # step2: for text, compute the corresponding text features in tensor [B, 10, 256]
        # select text_feat of given bbox/ object_tokens
        text_feat = text_feats[:, index]
        # step3: compute intra_sample_loss and inter_sample_loss
        if loss is None:
            loss = self.criterion(sim_matrix(text_feat, vid_feat))
        else:
            loss += self.criterion(sim_matrix(text_feat, vid_feat))
        return loss