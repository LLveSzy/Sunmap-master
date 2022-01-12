import torch
import tifffile
from utils import *
from tqdm import tqdm
from models.losses import *
from models import create_model
from multiprocessing import Pool
from torch.autograd import Variable
from options.val_options import ValOptions


def eval_net(model, testloader, device, n_val):
    junk_rat = 0
    global_steps = 0
    total_loss_iou = 0
    total_loss_tiou = 0
    cl_recall, cl_acc = 0, 0

    model.eval()
    with tqdm(total=n_val, desc='Validation round', unit='batch') as pbar:
        for batch, (data, label) in enumerate(testloader):
            data = Variable(data.to(device))
            label = Variable(label.clone().to(device))
            with torch.no_grad():
                pre = model(data)
                pre = torch.sigmoid(pre)
            pre[pre >= 0.5] = 1
            pre[pre < 0.5] = 0
            label[label > 0] = 1
            tifffile.imsave('./predict/' + str(batch) + '.tiff', pre.cpu().numpy()[0][0])
            tifffile.imsave('./predict/' + str(batch) + '_label.tiff', label.cpu().numpy()[0][0])
            tifffile.imsave('./predict/' + str(batch) + '_data.tiff', data.cpu().numpy()[0][0])

            recall, acc = soft_cldice_f1(pre, label)
            cl_recall += recall.cpu()
            cl_acc += acc.cpu()

            total_loss_iou += iou(pre, label).cpu()
            junk_rat += junk_ratio(pre, label).cpu()
            total_loss_tiou += t_iou(pre, label).cpu()

            global_steps += 1
            pbar.update(data.shape[0])
    model.train()
    cl_acc_mean = cl_acc / global_steps
    cl_recall_mean = cl_recall / global_steps

    return {'cl_acc': cl_acc_mean,
            'cl_recall': cl_recall_mean,
            'iou': total_loss_iou / global_steps,
            'tiou': total_loss_tiou / global_steps,
            'junk_ratio': junk_rat / global_steps,
            'cldice': (2. * cl_recall_mean * cl_acc_mean) / (cl_recall_mean + cl_acc_mean)}


if __name__ == '__main__':
    opt = ValOptions().parse()
    model = create_model(opt)
    val_type = opt.val_type
    if val_type == 'volumes2':
        model.eval_two_volumes_maxpool()
    elif val_type == 'cubes':
        model.eval_volumes_batch()
    elif val_type == 'segment':
        imgs = model.test_3D_volume()
        pool = Pool(processes=model.opt.process)
        pool.map(model.segment_brain_batch, imgs)
        pool.close()
        pool.join()


