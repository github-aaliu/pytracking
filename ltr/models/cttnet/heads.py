

from torch import nn
from ltr.models.target_classifier.initializer import FilterPool
from ltr.models.layers.filter import apply_filter
class Ctt_head(nn.Module):

    def __init__(self, kernel_size = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.filter_pool = FilterPool(filter_size = self.kernel_size)
    def forward(self, train_feat, test_feat, train_bb):

        kernel = self.filter_pool(train_feat, train_bb)
        scores = apply_filter(test_feat, kernel)
        return scores




