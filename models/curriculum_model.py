import os
import time
import torch
import logging
import torch.nn as nn
from . import networks
from data import create_dataset
from .base_model import BaseModel
from torch.autograd import Variable


class Curriculum(BaseModel):
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
            self.criterion_BCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(self.device))  # define loss.
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                                        patience=12, factor=0.5, threshold=0.001)

    def backward(self, pre, label):
        self.loss = self.criterion_BCE(pre, label)
        self.loss.backward()

    def forward(self, input):
        return self.model(input)

    def optimize_parameters(self, data):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        data[0] = Variable(data[0].float().to(self.device))
        data[0].requires_grad = True
        label = Variable(data[1].to(self.device))
        pre = self.forward(data[0])
        self.optimizer.zero_grad()
        self.backward(pre, label)
        self.optimizer.step()



