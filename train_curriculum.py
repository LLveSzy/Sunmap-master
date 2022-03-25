import time
import torch
import random
import  numpy as np
from tqdm import tqdm
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions

if __name__ == '__main__':
    # seed = 1343
    seed = 1268
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    opt = TrainOptions().parse()   # get training options
    dataset_train = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_train_size = len(dataset_train)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_train_size)
    opt.isTrain = False
    opt.dataset_mode = 'axons'
    dataset_val = create_dataset(opt)
    dataset_val_size = len(dataset_val)
    opt.isTrain = True

    model = create_model(opt)      # create a model given opt.model and other options
    total_iters = 0                # the total number of training iterations
    for epoch in range(1, opt.n_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        # model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        with tqdm(total=dataset_train_size, desc=f'Epoch {epoch}/{opt.n_epochs}', unit='volumes') as pbar:
            for i, data in enumerate(dataset_train):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.optimize_parameters(data)  # calculate loss functions, get gradients, update network weights
                if epoch % 2 == 0:
                    dataset_train.dataset.update(1e-2, data[0].grad.sign(), data[2])
                pbar.update(data[0].shape[0])
                pbar.set_postfix(**{'loss (batch)': format(model.loss.item(), '.5f')})

        model.eval_net(dataset_val, dataset_val_size)


