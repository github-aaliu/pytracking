import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from ltr.DCNv2.functions.deform_conv_func import DeformConvFunction
class DeformConv2d(nn.Module):
    def __init__(self, pad=True, stride=1):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.padding = pad
        self.stride = stride


    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x,target_filer,offset,feat_group=False):

        dtype = x.dtype
        ks = target_filer.size(-1)

        N = ks**2

        num_img,num_sq, ct, h, w = x.size()
        x = x.view(-1, *x.shape[2:])

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N,ks, dtype).to(x.device)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, ks//2, N, dtype).to(x.device)
        p_0 = p_0.expand(offset.size(0),-1,-1,-1)
        p_n = torch.cat([p_n[:,:N,...]*ks/offset[:,0].view(-1,1,1,1), p_n[:, N:,...]*ks/offset[:,1].view(-1,1,1,1)], dim=1)
        p = p_0+p_n

        if self.padding:
            zero_padding = nn.ZeroPad2d(ks // 2)
            x = zero_padding(x)

        # (b, h, w, 2N)
        p = p.permute(0, 2, 3, 1).to(x.device)


        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if not feat_group:
            x_offset = x_offset.view(-1, num_sq*ct, h, w, N)
            x_offset = self._reshape_x_offset(x_offset, ks)

            out = F.conv2d(x_offset, target_filer, stride=ks, groups=num_sq)
            return out

        else:
            x_offset = x_offset.view(-1, num_sq, ct, h, w, N)
            x_offset = self._reshape_grop_x_offset(x_offset)
            target_filer = target_filer.view(*target_filer.shape[:2], -1).permute(0, 2, 1)
            target_filer = target_filer.contiguous().view(-1, target_filer.size(-1), 1, 1)

            out = F.conv2d(x_offset, target_filer, groups=N * num_sq)

            Dconv_feat = out.view(-1, N, h, w)
            return Dconv_feat


    def _get_p_n(self, N, kernel_size, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(kernel_size-1)//2, (kernel_size-1)//2+1),
            torch.arange(-(kernel_size-1)//2, (kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, padding,  N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(padding, h+padding),
            torch.arange(padding, w+padding))
        p_0_x = p_0_x.view(1, 1, h, w).expand(-1, N, -1, -1)
        p_0_y = p_0_y.view(1, 1, h, w).expand(-1, N, -1, -1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0


    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        x_b, c = x.size(0), x.size(1)

        # (b, c, h*w)
        x = x.view(x_b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        index = index.reshape(x_b,-1,h,w,N).permute(0,2,3,1,4).reshape(x_b,1,h,w,-1)

        # (b, c, h*w*N)
        index = index.expand(-1, c, -1, -1, -1).reshape(x_b, c, -1)

        x_offset = x.gather(dim=-1, index=index).reshape(x_b, c, h, w, -1, N)
        x_offset = x_offset.permute(0,4,1,2,3,5).reshape(-1,c,h,w,N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].reshape(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.reshape(b, c, h*ks, w*ks)

        return x_offset

    @staticmethod
    def _reshape_grop_x_offset(x_offset):
        m,b, c, h, w, N = x_offset.size()
        x_offset = x_offset.permute(0,1,5,2,3,4)
        x_offset = x_offset.reshape(m, -1, h, w)
        return x_offset



class DCNv2(nn.Module):
    def __init__(self, kernel_size=3, padding=1, dilation=1, stride=1, im2col_step=64):
        super(DCNv2, self).__init__()
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.im2col_step = im2col_step
        self.get_p_n()

    def get_p_n(self):
        kernel_size = self.kernel_size
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(kernel_size-1)//2, (kernel_size-1)//2+1),
            torch.arange(-(kernel_size-1)//2, (kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        self.p_n = p_n.view(1, 1, 2*kernel_size*kernel_size)


    def forward(self, input, weight, offset):
        dtype = input.dtype
        grouds,_,_,ks = weight.size()
        weight = weight.permute(2,3,0,1).contiguous()
        N = ks**2
        off_b = offset.size(0)
        b_num, num_sq, ct, h, w = input.size()
        num_p = int(off_b/b_num)
        if num_p > 1:
            input = input.view(b_num,1,num_sq, ct, h, w).expand(-1,num_p,-1,-1,-1,-1)
            input = input.reshape(-1, num_sq, ct, h, w)
        input = input.contiguous().view(-1, num_sq*ct, h, w)

        p_n = self.p_n.clone()
        p_n = p_n.expand(off_b, grouds, -1).type(dtype).to(input.device)
        offset_t = torch.cat((p_n[...,:N] * ks / offset[..., 0].unsqueeze(-1),
                         p_n[...,N:] * ks / offset[..., 1].unsqueeze(-1)), dim=2).contiguous()
        return DeformConvFunction.apply(input, offset_t,
                                        weight,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        grouds,
                                        self.im2col_step)