import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.models.layers.blocks import conv_block


class ResponsePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg_centi = 0.2
        self.bg_conf = 0
        self.dimp_thresh = 0.05


    def forward(self, cost_volume, state_prev, dimp_score_cur, init_label=None,  prev_scores_dimp=None, dimp_thresh=None,
                output_window=None):
        # Cost vol shape: n x h*w x h x w
        # state_prev shape: n x d x h x w
        # dimp_cur_shape: n x 1 x h x w
        # init_label shape: n x 1 x h x w
        if dimp_thresh is None:
            dimp_thresh = self.dimp_thresh
        auxiliary_outputs = {}

        num_sequences = cost_volume.shape[0]
        feat_sz = cost_volume.shape[-2:]

        cost_volume_p1 = cost_volume.view(-1, feat_sz[0] * feat_sz[1])
        cost_volume_p1 = F.softmax(cost_volume_p1, dim=1)

        cost_volume_p2 = cost_volume_p1.view(num_sequences, -1, feat_sz[0], feat_sz[1])
        cost_volume_p2 = F.softmax(cost_volume_p2, dim=1)

        auxiliary_outputs['cost_volume_processed'] = cost_volume_p2


        if state_prev is None:
            state_new = init_label
        else:
            if prev_scores_dimp.max()<0.2:
                state_new = torch.zeros(state_prev.shape).to(state_prev.device)
            else:
                bg_thr = prev_scores_dimp.max() * self.bg_centi
                state_new = torch.where(init_label>0.1, init_label, -(prev_scores_dimp > bg_thr).float())

        state_prev_ndhw = state_new.view(num_sequences, -1, feat_sz[0], feat_sz[1])

        state_prev_nhwd = state_prev_ndhw.permute(0, 2, 3, 1).contiguous(). \
            view(num_sequences, feat_sz[0] * feat_sz[1], -1, 1, 1).expand(-1, -1, -1, feat_sz[0], feat_sz[1])

        #  Compute propagation weights
        propagation_weight_norm = cost_volume_p2.view(num_sequences, feat_sz[0] * feat_sz[1], 1, feat_sz[0], feat_sz[1])


        propagation_conf = propagation_weight_norm.view(num_sequences, -1, feat_sz[0], feat_sz[1])
        propagation_conf = -(propagation_conf * (propagation_conf + 1e-4).log()).sum(dim=1)

        auxiliary_outputs['propagation_weights'] = propagation_weight_norm

        propagated_h = (propagation_weight_norm * state_prev_nhwd).sum(dim=1)
        propagated_h = propagated_h.view(-1, feat_sz[0]*feat_sz[1])
        propagated_h = F.softmax(propagated_h, dim=1)
        propagated_h = propagated_h.view(num_sequences, -1, feat_sz[0], feat_sz[1])
        auxiliary_outputs['propagated_h'] = propagated_h.clone()


        propagation_conf = propagation_conf.view(num_sequences, 1, feat_sz[0], feat_sz[1])
        auxiliary_outputs['propagation_conf'] = propagation_conf
        fused_prediction = propagated_h    #need change


        auxiliary_outputs['fused_score_orig'] = fused_prediction.clone()
        if dimp_thresh is not None:
            fused_prediction = fused_prediction * (dimp_score_cur > dimp_thresh).float()

        fused_prediction = dimp_score_cur + self.bg_conf * fused_prediction


        return fused_prediction, state_new, auxiliary_outputs

