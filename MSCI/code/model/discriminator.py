import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
#判别器输入为属性特征对象特征 ,输出为是否为原始组合的概率
class CombinationDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(CombinationDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # 判别是原始组合
            nn.Sigmoid()
        )

    def forward(self, attr_features, obj_features):
        combined_features = torch.cat([attr_features, obj_features], dim=-1)
        return self.fc(combined_features)
