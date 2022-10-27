# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class VAM(nn.Module):
    """ Visual Aware Model"""
    def __init__(self, base_model, feature_dim):
        super(VAM, self).__init__()
        self.features = base_model
        self.feature_dim = feature_dim

    def forward(self, images):
        feats = self.features(images)
        feats = feats.view(-1, self.feature_dim)
        out = torch.mean(feats, -1)
        return out


class TestVAM(nn.Module):
    """ Visual Aware Model for Inference"""
    def __init__(self, base_model, feature_dim):
        super(TestVAM, self).__init__()
        self.features = base_model
        self.feature_dim = feature_dim

    def forward(self, images):
        feats = self.features(images)
        feats = feats.view(feats.size(0), 1, -1)
        scores = nn.functional.avg_pool1d(feats, kernel_size=self.feature_dim, stride=self.feature_dim)
        scores = scores.view(-1)
        return feats, scores


