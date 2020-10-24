import torch
import torch.nn as nn


class RegModel(nn.Module):


    def __init__(self, target_size):
        super().__init__()
        self.fc = nn.Sequential(

            nn.Conv2d(target_size**2, 32,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4,
                      kernel_size=3, stride=1,
                      padding=1, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,Dconv_feat,num_sq,h,w):
        scores = [f.sum(2) for f in Dconv_feat]
        reg_out=[]
        for f in Dconv_feat:
            f = f.view(-1, 9, h, w)
            reg_out.append((self.fc(f)).view(-1, num_sq, 4, h, w)) #need add softmax or not?========
        return scores, reg_out