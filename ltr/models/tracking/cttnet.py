import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor
from ltr.models.backbone.hourglass import HourglassNet
from ltr.models.cttnet.heads import Ctt_head

class CttNet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, heads):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.head = heads


    def forward(self, train_imgs, test_imgs, train_bb):

        # Extract backbone features
        train_feat = self.feature_extractor(train_imgs)
        test_feat = self.feature_extractor(train_imgs)


        # Run heads module
        target_scores = self.head(train_feat, test_feat, train_bb)

        return target_scores




@model_constructor
def cttnet(n_stack=1, kernel_size=4):
    # Backbone
    backbone_net = HourglassNet(n_stack)

    head = Ctt_head(kernel_size = kernel_size)
    # DiMP network
    net = CttNet(feature_extractor=backbone_net, heads=head)
    return net

