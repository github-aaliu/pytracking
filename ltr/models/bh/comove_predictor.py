from torch import nn
from ltr.models.layers.blocks import conv_block
import numpy as np
import torch
from ltr.models.kys.utils import shift_features
import math
import torch.nn.functional as F
from pytracking.libs.dcf import label_function_spatial
from pytracking.libs.dcf import gauss_map

class comove_predictor(nn.Module):
    def __init__(self, cost_volume=None, kalmanFilter=None):
        super().__init__()
        self.compute_cost_volume = cost_volume
        self.kalmanFilter = kalmanFilter
        self.conf_measure = 'entropy'
        self.bg_threshold = 7

        self.vel_para = 0.025
        """
        self.cost_volume_proc1 = nn.Sequential(
            conv_block(1, 8, kernel_size=3, stride=1, padding=1, batch_norm=True, relu=True),
            conv_block(8, 1, kernel_size=3, stride=1, padding=1, batch_norm=True, relu=False))

        self.cost_volume_proc2 = nn.Sequential(
            conv_block(1, 8, kernel_size=3, stride=1, padding=1, batch_norm=True, relu=True),
            conv_block(8, 1, kernel_size=3, stride=1, padding=1, batch_norm=True, relu=False))
        """
    def comp_cost_volume(self,feat1,feat2):

        feat_size = feat1.shape[2:]
        self.feat_size = feat_size
        num_sq = feat1.shape[0]
        cost_volume = self.compute_cost_volume(feat1, feat2)


        """
        cost_volume = cost_volume.view(-1,1,feat_size[0],feat_size[1])
        cost_volume_p1 = self.cost_volume_proc1(cost_volume).view(-1,feat_size[0]*feat_size[1])
        cost_volume_p1 = nn.Softmax(cost_volume_p1,dim=1)
        cost_volume_p1 = cost_volume_p1.view(-1,1,feat_size[0],feat_size[1])

        cost_volume_p2 = self.cost_volume_proc2(cost_volume_p1)
        cost_volume_p2 = cost_volume_p2.view(num_sq,-1,feat_size[0],feat_size[1])
        self.cost_volume_p2 = nn.Softmax(cost_volume_p2,dim=1)
        """
        """
        cost_volume = cost_volume.view(-1, 1, feat_size[0], feat_size[1])
        cost_volume_p1 = cost_volume.view(-1,feat_size[0]*feat_size[1])
        cost_volume_p1 = F.softmax(cost_volume_p1,dim=1)

        cost_volume_p2 = cost_volume_p1.view(num_sq,-1,feat_size[0],feat_size[1])
        cost_volume_p2 = F.softmax(cost_volume_p2,dim=1)
        """

        cost_volume_p2=cost_volume.view(num_sq, -1, feat_size[0], feat_size[1])
        self.cost_volume = cost_volume_p2.view(-1,feat_size[0],feat_size[1],feat_size[0],feat_size[1])

        self.propagation_conf = -(cost_volume_p2 * (cost_volume_p2 + 1e-4).log()).sum(dim=1)


    def get_bg_loc(self,obj_loc,obj_size):
        conf = self.propagation_conf.clone()
        propagation_conf = (torch.ones(conf.shape)*50).to(conf.device)
        feat_obj_loc = obj_loc-torch.tensor([0.5,0.5])
        obj_round = [math.floor(feat_obj_loc[0]-obj_size[0]/2),math.floor(feat_obj_loc[1]-obj_size[1]/2),
                   math.ceil(feat_obj_loc[0] + obj_size[0] / 2),math.ceil(feat_obj_loc[1]+obj_size[1]/2)]

        propagation_conf[:,3:-2,3:-2] = conf[:,3:-2,3:-2]
        propagation_conf[:,obj_round[0]:obj_round[2],obj_round[1]:obj_round[3]]=50
        if propagation_conf.view(-1).min()>self.bg_threshold:
            return None, None
        else:
            bg_min_id = propagation_conf.view(-1).argmin()
            now_bg_loc= torch.tensor([(bg_min_id//self.feat_size[0]).item(), (bg_min_id%self.feat_size[0]).item()])
            pre_bg_id = self.cost_volume[0,:,:,now_bg_loc[0],now_bg_loc[1]].view(-1).argmax()
            pre_bg_loc = torch.tensor([(pre_bg_id//self.feat_size[0]).item(), (pre_bg_id%self.feat_size[0]).item()])
            return now_bg_loc, pre_bg_loc




    def predictor(self,data):
        feat1 = data['feat1']
        feat2 = data['feat2']
        bg_loc = data['bg_loc']
        dimp_score_cur = data['dimp_score_cur']
        feat_size = feat1.shape[2:]
        feat_size_tensor = torch.tensor([feat_size[0],feat_size[1]])
        num_sq = feat1.shape[0]
        self.comp_cost_volume(feat1,feat2)


        bg_cost = self.cost_volume[0,bg_loc[0].item(),bg_loc[1].item(),:,:].view(-1)
        bg_conf = -(bg_cost * (bg_cost + 1e-4).log()).sum()


        move_state_np, vel_move_conf_np = self.kalmanFilter.predict()
        move_state = torch.from_numpy(move_state_np)
        vel_move_conf = torch.from_numpy(vel_move_conf_np).diag()
        move_conf = vel_move_conf[:2]*self.vel_para
        movepre_loc = move_state[:2]
        obj_max_id = dimp_score_cur.view(-1).argmax()
        obj_loc = torch.tensor([(obj_max_id//feat_size[0]).item(), (obj_max_id%feat_size[0]).item()])
        pre_bg_loc = obj_loc-movepre_loc
        g_center = pre_bg_loc-feat_size_tensor//2

        g_win = gauss_map(feat_size_tensor, move_conf, g_center).to(dimp_score_cur.device)
        g_id = g_win.view(-1).argmax()
        g_loc = torch.tensor([(g_id//feat_size[0]).item(), (g_id%feat_size[0]).item()])

        bg_cost_map = bg_cost.view(-1,1,feat_size[0],feat_size[1])
        bg_win_map = g_win*bg_cost_map
        obj2bg_val = bg_win_map.view(-1).max()+dimp_score_cur.view(-1).max()


        bg_max_loc = bg_cost_map.view(-1).argmax()

        s_bg_loc = torch.tensor([(bg_max_loc//feat_size[0]).item(), (bg_max_loc%feat_size[0]).item()])
        pre_obj_loc = s_bg_loc+movepre_loc
        o_center = pre_obj_loc - feat_size_tensor // 2
        o_win = gauss_map(feat_size_tensor, move_conf, o_center).to(dimp_score_cur.device)

        obj_win_map = o_win*dimp_score_cur

        bg2obj_val = obj_win_map.view(-1).max()+bg_cost_map.view(-1).max()

        if bg2obj_val>=obj2bg_val:
            new_bg_loc = s_bg_loc
            new_obj_max = obj_win_map.view(-1).argmax()
            new_obj_loc = torch.tensor([(new_obj_max//feat_size[0]).item(), (new_obj_max%feat_size[0]).item()])
        else:
            new_obj_loc = obj_loc
            new_bg_max = bg_win_map.view(-1).argmax()
            new_bg_loc = torch.tensor([(new_bg_max//feat_size[0]).item(), (new_bg_max%feat_size[0]).item()])

        obj__max = dimp_score_cur.view(-1).max()

        return obj_loc,s_bg_loc, 0, obj__max.item(), 'normal'








