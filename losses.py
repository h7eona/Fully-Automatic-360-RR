import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

from einops import rearrange, repeat




def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
    
class VisualizeAttnMap(nn.Module):
    def __init__(self, H, W, head_fusion):
        super(VisualizeAttnMap, self).__init__()
        self.resize = torchvision.transforms.Resize((H, W), interpolation=Image.BICUBIC)
        self.head_fusion = head_fusion
        self.H = H
        self.W = W

    def forward(self, attn_list):
        head_attns = []
        # joint_attentions = []

        attn_list = torch.mean(attn_list, dim=2) #Map mean # nw heads h w
        attn_list = rearrange(attn_list, 'nW heads w -> heads nW w')
        attn_list = repeat(attn_list, 'heads h w -> heads c h w', c=3)

        for head_num in range(attn_list.shape[0]):
            head_attn = attn_list[head_num, :, : ,:]
            resized = self.resize(head_attn)
            resized = resized / torch.max(resized)
            resized = torch.clamp(resized,0,1)
            head_attns.append(resized)
        return head_attns
