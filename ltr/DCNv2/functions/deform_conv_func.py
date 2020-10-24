#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

import DCN

class DeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight,
                stride, padding, dilation, group, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[:2])
        ctx.group = group
        ctx.im2col_step = im2col_step
        output = DCN.deform_conv_forward(input, weight, offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.group,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input, grad_weight = \
            DCN.deform_conv_backward(input, weight,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.group,
                                     ctx.im2col_step)

        return grad_input, grad_weight,\
            None, None, None, None, None, None, None,None
