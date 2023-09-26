import torch.nn as nn
from lib.utils import net_utils
import torch
from lib.train.trainers import focal


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.vote_crit = torch.nn.functional.smooth_l1_loss
        # self.seg_crit = nn.CrossEntropyLoss()
        self.seg_focal = focal.FocalLoss()



    def forward(self, batch):
        output = self.net(batch['inp'])

        scalar_stats = {}
        loss = 0

        if 'pose_test' in batch['meta'].keys():
            loss = torch.tensor(0).to(batch['inp'].device)
            return output, loss, {}, {}

        weight = batch['mask'][:, None].float()

        vote_loss = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
        # print("预测和真实vector, 权重:", output['vertex'].sum(), batch['vertex'].sum(), weight.sum())
        # print("初始vote_loss, weight_sum", vote_loss, weight.sum())
        vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)

        scalar_stats.update({'vote_loss': vote_loss})
        loss += vote_loss

        mask = batch['mask'].long()
        # print("真实分割和预测分割", mask.sum(), output['seg'].sum())
        # seg_loss = self.seg_crit(output['seg'], mask)
        seg_loss = self.seg_focal(output['seg'], mask)#我的损失
        # print("seg_loss",seg_loss)
        scalar_stats.update({'seg_loss': seg_loss})
        loss += seg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
