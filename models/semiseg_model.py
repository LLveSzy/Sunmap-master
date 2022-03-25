import os
import time
import torch
import logging
import torch.nn as nn
from . import networks
from utils import torch_fliplr
from .base_model import BaseModel
from torch.autograd import Variable


class SemiSeg(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.metric = 0
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.model = networks.define_net(opt.input_nc, opt.output_nc, opt.net, gpu_ids=self.gpu_ids)
        self.ema_model = networks.define_net(opt.input_nc, opt.output_nc, opt.net, gpu_ids=self.gpu_ids)
        for param in self.ema_model.parameters():
            param.detach_()

        if opt.pre_trained:
            pretrain_encoder = torch.load(opt.pre_trained, map_location=self.device)
            self.model.load_state_dict(networks.load_my_state_dict(self.model, pretrain_encoder))
            print(f'loaded: {opt.pre_trained}')

        if self.isTrain:
            self.criterion_BCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(self.device))  # define loss.
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 3]).to(self.device), ignore_index=255)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                                        patience=20, factor=0.5, threshold=0.001)

    def backward(self, pre, ema_pre, label):
        if self.opt.output_nc == 1:
            pre_soft = torch.sigmoid(pre)
            ema_soft = torch.sigmoid(ema_pre)
            supervised_loss = self.criterion_BCE(pre[:self.opt.labeled_bs], label[:self.opt.labeled_bs])  # + \
            # 0.2 * (1 + soft_cldice_loss(torch.sigmoid(pre[:labeled_bs]), label[:labeled_bs]))
            consistency_loss = torch.mean((pre_soft[self.opt.labeled_bs:] - ema_soft) ** 2)
            self.loss = supervised_loss + 0.7 * consistency_loss #- 0.2 * (pre_soft * torch.log(pre_soft + 1e-9)).mean()
        else:
            pre_soft = torch.softmax(pre, dim=1)
            ema_soft = torch.softmax(ema_pre, dim=1)
            ema_label = torch.argmax(ema_soft, dim=1)
            ema_label[ema_label == 0] = 255
            ema_label[torch.max(ema_soft, 1)[0] < 0.7] = 255

            label[self.opt.labeled_bs:] = ema_label
            supervised_loss = self.criterion(pre, label)
            consistency_loss = torch.mean((pre_soft[self.opt.labeled_bs:] - ema_soft) ** 2)
            self.loss = supervised_loss + 0.5 * consistency_loss - 0.1 * (pre_soft * torch.log(pre_soft + 1e-9)).mean()
        self.loss.backward()

    def forward(self, input):
        return self.model(input)

    def optimize_parameters(self, data):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        input = Variable(data[0].to(self.device))
        label = Variable(data[1].to(self.device))
        pre = self.forward(input)
        unlabeled_volume_batch = input[self.opt.labeled_bs:]
        noise = torch.clamp(torch.randn_like(
            unlabeled_volume_batch) * 0.05, -0.05, 0.05)
        ema_inputs = torch_fliplr(unlabeled_volume_batch + noise)
        with torch.no_grad():
            ema_pre = torch_fliplr(self.ema_model(ema_inputs))

        self.optimizer.zero_grad()
        self.backward(pre, ema_pre, label)
        self.optimizer.step()

    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(semi='True')
        if is_train:
            parser.add_argument('--labeled_bs', type=int, default=15, help='weight for L1 loss')
        return parser


    def update_ema_variables(self, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
