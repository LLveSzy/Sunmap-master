import torch
import tifffile
from utils import *
from tqdm import tqdm
from . import networks
from models.losses import *
from data import create_dataset
from torch.autograd import Variable


class EvalNet:
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')


    def eval_two_volumes_maxpool(self):
        pre = read_tiff_stack(self.opt.dataroot)
        label = read_tiff_stack(self.opt.data_target)
        kernel = (self.opt.pool_kernel, self.opt.pool_kernel, self.opt.pool_kernel)
        pre[pre < self.opt.threshold] = 0
        pre[pre >= self.opt.threshold] = 1
        label[label > 0] = 1
        pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(self.device)
        label = torch.Tensor(label).view((1, 1, *label.shape)).to(self.device)
        pre = torch.nn.functional.max_pool3d(pre, kernel, kernel, 0)
        label = torch.nn.functional.max_pool3d(label, kernel, kernel, 0)

        dice_score = dice_error(pre, label)

        total_loss_iou = iou(pre, label).cpu()
        total_loss_tiou = t_iou(pre, label).cpu()
        recall, acc = soft_cldice_f1(pre, label)
        cldice = (2. * recall * acc) / (recall + acc)

        print('\n Validation IOU: {}\n T-IOU: {}'
              '\n ClDice: {} \n ClAcc: {} \n ClRecall: {} \n Dice-score: {}'
              .format(total_loss_iou, total_loss_tiou, cldice, acc, recall, dice_score, '.8f'))

    def eval_volumes_batch(self):
        model = networks.define_net(self.opt.input_nc, self.opt.output_nc, self.opt.net, gpu_ids=self.gpu_ids)
        if self.opt.pre_trained:
            pretrain_encoder = torch.load(self.opt.pre_trained, map_location=self.device)
            model.load_state_dict(networks.load_my_state_dict(model, pretrain_encoder))
            print(f'loaded: {self.opt.pre_trained}')
        testLoader = create_dataset(self.opt)
        n_val = len(testLoader)
        loss_dir = self.eval_net(model, testLoader, self.device, n_val)
        iou, t_iou = loss_dir['iou'], loss_dir['tiou']
        cldice, clacc, clrecall = loss_dir['cldice'], loss_dir['cl_acc'], loss_dir['cl_recall']
        junk_rat = loss_dir['junk_ratio']
        print('\n Validation IOU: {}\n T-IOU: {}'
              '\n ClDice: {} \n ClAcc: {} \n ClRecall: {}'
              '\n Junk-ratio: {}'
              .format(iou, t_iou, cldice, clacc, clrecall, junk_rat, '.8f'))

    @staticmethod
    def eval_net(model, testloader, device, n_val):
        total_loss_iou = 0
        total_loss_tiou = 0
        junk_rat = 0
        cl_recall, cl_acc = 0, 0
        global_steps = 0
        model.eval()
        with tqdm(total=n_val, desc='Validation round', unit='batch') as pbar:
            for batch, (data, label) in enumerate(testloader):
                data = Variable(data.to(device))
                label = Variable(label.clone().to(device))

                with torch.no_grad():
                    pre = model(data)
                    pre = torch.sigmoid(pre)
                # tifffile.imsave(data_root +'/predict/' + str(batch) + '.tiff', pre.cpu().numpy()[0][0])
                # tifffile.imsave(data_root + '/predict/' + str(batch) + '_label.tiff', label.cpu().numpy()[0][0])
                # tifffile.imsave(data_root + '/predict/' + str(batch) + '_data.tiff', data.cpu().numpy()[0][0])
                pre[pre > 0.5] = 1
                pre[pre <= 0.5] = 0
                label[label > 0.5] = 1
                label[label <= 0.5] = 0

                total_loss_iou += iou(pre, label).cpu()
                total_loss_tiou += t_iou(pre, label).cpu()
                junk_rat += junk_ratio(pre, label).cpu()
                recall, acc = soft_cldice_f1(pre, label)
                cl_recall += recall.cpu()
                cl_acc += acc.cpu()
                global_steps += 1
                pbar.update(data.shape[0])
        model.train()
        cl_recall_mean = cl_recall / global_steps
        cl_acc_mean = cl_acc / global_steps
        return {'iou': total_loss_iou / global_steps,
                'cldice': (2. * cl_recall_mean * cl_acc_mean) / (cl_recall_mean + cl_acc_mean),
                'cl_acc': cl_acc_mean,
                'cl_recall': cl_recall_mean,
                'tiou': total_loss_tiou / global_steps,
                'junk_ratio': junk_rat / global_steps}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(semi='False')
        val_type = parser.parse_known_args()[0].val_type
        if val_type == 'cubes':
            parser.add_argument('--net', type=str, help='unet | axialunet')
            parser.add_argument('--batch_size', type=int, default=10, help='batch size')
            parser.add_argument('--n_samples', type=int, default=4, help='crop n volumes from one cube')
            parser.add_argument('--artifacts', type=bool, default=True, help='train with artifacts volumes')
            parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model')
            parser.add_argument('--shuffle_val', type=bool, default=False, help='whether to shuffle the val data')
        elif val_type == 'volumes2':
            parser.add_argument('--data_target', type=str, help='target volume for evaluating')
            parser.add_argument('--pool_kernel', type=int, default=5, help='maxpooling kernel size')

        return parser