import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        ignore_index=255,
) -> torch.Tensor:

    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype)

    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret


#实际计算的损失函数,降低非目标训练权重的同时提升目标的训练权重
#解决样本空间中目标和非目标所占比重不均匀的问题
def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0)  # B

    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]

    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)

        # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + eps
    #编码各个真实值,
    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
                                            dtype=input.dtype, ignore_index=ignore_index)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)
    # print("训练权重:",weight)
    #print("权重形状:",weight.shape)
    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)
    #print("焦点focal",focal.shape)
    # loss_tmp : (B, H, W)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    #总共2039个数据
    # np.set_printoptions(threshold=np.inf)
    # torch.set_printoptions(profile='full')
    # #背景
    # fp1 = open('/home/ivclab/homework/vis/2d_weight1.txt', 'a')
    # fp2 = open('/home/ivclab/homework/vis/2d_weight2.txt', 'a')
    # fp3 = open('/home/ivclab/homework/vis/2d_weight3.txt', 'a')
    # fp4 = open('/home/ivclab/homework/vis/2d_weight4.txt', 'a')
    # fp5 = open('/home/ivclab/homework/vis/2d_weight5.txt', 'a')
    # fp6 = open('/home/ivclab/homework/vis/2d_weight6.txt', 'a')
    # fp7 = open('/home/ivclab/homework/vis/2d_weight7.txt', 'a')
    # fp8 = open('/home/ivclab/homework/vis/2d_weight8.txt', 'a')
    # fp9 = open('/home/ivclab/homework/vis/2d_weight9.txt', 'a')
    # fp10 = open('/home/ivclab/homework/vis/2d_weight10.txt', 'a')
    # #目标
    # fp11 = open('/home/ivclab/homework/vis/2d_weight11.txt', 'a')
    # fp12 = open('/home/ivclab/homework/vis/2d_weight12.txt', 'a')
    # fp13 = open('/home/ivclab/homework/vis/2d_weight13.txt', 'a')
    # fp14 = open('/home/ivclab/homework/vis/2d_weight14.txt', 'a')
    # fp15 = open('/home/ivclab/homework/vis/2d_weight15.txt', 'a')
    # fp16 = open('/home/ivclab/homework/vis/2d_weight16.txt', 'a')
    # fp17 = open('/home/ivclab/homework/vis/2d_weight17.txt', 'a')
    # fp18 = open('/home/ivclab/homework/vis/2d_weight18.txt', 'a')
    # fp19 = open('/home/ivclab/homework/vis/2d_weight19.txt', 'a')
    # fp20 = open('/home/ivclab/homework/vis/2d_weight20.txt', 'a')
    # #
    # #
    # #
    # #
    # #
    # np_weight = weight.detach().cpu().numpy()
    # # print("取出来的像素值变化:",np_weight[0][1][0][0])
    # fp1.write(str(np_weight[0][0][0][0]))
    # fp1.write(",")
    # fp2.write(str(np_weight[0][0][0][11]))
    # fp2.write(",")
    # fp3.write(str(np_weight[0][0][0][22]))
    # fp3.write(",")
    # fp4.write(str(np_weight[0][0][0][33]))
    # fp4.write(",")
    # fp5.write(str(np_weight[0][0][0][44]))
    # fp5.write(",")
    # fp6.write(str(np_weight[0][0][0][55]))
    # fp6.write(",")
    # fp7.write(str(np_weight[0][0][0][66]))
    # fp7.write(",")
    # fp8.write(str(np_weight[0][0][0][77]))
    # fp8.write(",")
    # fp9.write(str(np_weight[0][0][0][88]))
    # fp9.write(",")
    # fp10.write(str(np_weight[0][0][0][29]))
    # fp10.write(",")
    #
    # fp11.write(str(np_weight[0][1][0][0]))
    # fp11.write(",")
    # fp12.write(str(np_weight[0][1][0][21]))
    # fp12.write(",")
    # fp13.write(str(np_weight[0][1][0][32]))
    # fp13.write(",")
    # fp14.write(str(np_weight[0][1][0][34]))
    # fp14.write(",")
    # fp15.write(str(np_weight[0][1][0][54]))
    # fp15.write(",")
    # fp16.write(str(np_weight[0][1][0][57]))
    # fp16.write(",")
    # fp17.write(str(np_weight[0][1][0][69]))
    # fp17.write(",")
    # fp18.write(str(np_weight[0][1][0][87]))
    # fp18.write(",")
    # fp19.write(str(np_weight[0][1][0][18]))
    # fp19.write(",")
    # fp20.write(str(np_weight[0][1][0][69]))
    # fp20.write(",")
    # #
    # fp1.close()
    # fp2.close()
    # fp3.close()
    # fp4.close()
    # fp5.close()
    # fp6.close()
    # fp7.close()
    # fp8.close()
    # fp9.close()
    # fp10.close()
    #
    # fp11.close()
    # fp12.close()
    # fp13.close()
    # fp14.close()
    # fp15.close()
    # fp16.close()
    # fp17.close()
    # fp18.close()
    # fp19.close()
    # fp20.close()

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):


    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8, ignore_index=30):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index


    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)
