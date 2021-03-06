import os
import time
import torch
import logging
import torch.nn as nn
from . import networks
from data import create_dataset
from .base_model import BaseModel
from torch.autograd import Variable
from models.losses import soft_cldice_loss


class SegNet(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.metric = 0
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.model = networks.define_net(opt.input_nc, opt.output_nc, opt.net, gpu_ids=self.gpu_ids)
        if opt.pre_trained:
            pretrain_encoder = torch.load(opt.pre_trained, map_location=self.device)
            self.model.load_state_dict(networks.load_my_state_dict(self.model, pretrain_encoder))
            print(f'loaded: {opt.pre_trained}')

        if self.isTrain:
            if self.opt.output_nc == 1:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(self.device))  # define loss.
            else:
                self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 3]).to(self.device), ignore_index=255)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                                        patience=12, factor=0.5, threshold=0.001)

    def backward(self, pre, label):
        # junks = label.view(label.shape[0], label.shape[1], -1).sum(axis=2)
        # junks = junks.view(pre.shape[0], pre.shape[1], 1, 1, 1).expand(pre.shape)
        # rect_label = (torch.sigmoid(pre) >= 0.80)
        # label = (label + rect_label - label * rect_label) * (junks != 0)
        self.loss = self.criterion(pre, label)
        # if self.opt.input_nc == 1:
        #     self.loss = self.loss + 0.05 * soft_cldice_loss(pre, label)
        self.loss.backward()

    def forward(self, input):
        return self.model(input)

    def optimize_parameters(self, data):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        input = Variable(data[0].to(self.device))
        label = Variable(data[1].to(self.device))
        pre = self.forward(input)
        self.optimizer.zero_grad()
        self.backward(pre, label)
        self.optimizer.step()


