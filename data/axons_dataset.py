import random
import tifffile
import itertools
import torch.utils.data as data

from tqdm import tqdm
from skimage import exposure
from utils import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from skimage.exposure import match_histograms


class AxonsDataset(data.Dataset):
    def __init__(self, opt):
        data_path = opt.dataroot
        n_samples = opt.n_samples
        input_dim = opt.input_dim
        data_mix = opt.artifacts
        in_channel = opt.input_nc
        unlabel = opt.semi
        self.flag = opt.isTrain

        self.datas = []
        self.labels = []
        self.labels_ori = []

        data_path = join(data_path, 'train') if self.flag else join(data_path, 'val')

        volumes_folder_path = join(data_path, "volumes")
        labels_folder_path = join(data_path, "labels_sk") if self.flag else join(data_path, 'labels')

        volumes_path = get_dir(volumes_folder_path)
        labels_path = get_dir(labels_folder_path)

        assert len(labels_path) == len(volumes_path)
        if n_samples == None:
            n_samples = len(labels_path)

        callfunc = {
            0: lambda: [np.fliplr(volume_rot), np.fliplr(label_rot)],
            1: lambda: [np.flipud(volume_rot), np.flipud(label_rot)],
         }

        total_volumes = 0
        with tqdm(total=len(volumes_path) * n_samples, desc=f'volumes numbers') as pbar:
            for vpath, lpath in zip(volumes_path, labels_path):
                # assert (vpath.split('/')[-1].replace('volume', 'label')) == lpath.split('/')[-1]
                volume = read_tiff_stack(vpath)
                label = read_tiff_stack(lpath)
                if volume.shape[0] < opt.input_dim or volume.shape[1] < opt.input_dim \
                        or volume.shape[2] < opt.input_dim:
                    continue
                for _ in range(n_samples):
                    z = random.randint(0, label.shape[0] - input_dim)
                    x = random.randint(0, label.shape[1] - input_dim)
                    y = random.randint(0, label.shape[2] - input_dim)
                    volume_chunk = volume[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    label_chunk = label[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    if random.randint(0, 1) == 0:
                        volume_chunk = contrast_augmentation(volume_chunk, label_chunk, rad=8, N=3)

                    k_seed = random.randint(0, 3)
                    flip_seed = random.randint(0, 1)

                    volume_rot = np.rot90(np.swapaxes(volume_chunk, 0, 2), k=k_seed).swapaxes(2, 0)
                    label_rot = np.rot90(np.swapaxes(label_chunk, 0, 2), k=k_seed).swapaxes(2, 0)

                    data, annotation = callfunc[flip_seed]()

                    # data_ori = data.copy()[np.newaxis, :, :, :]
                    data_ori = ((data - data.min()) / (data.max() - data.min()))[np.newaxis, :, :, :]
                    # data_ori = equal(data, 0.9)[np.newaxis, :, :, :]
                    if in_channel != 1:
                        im = equal(gauss_cal(data), 0.1)
                        data[im < 0.1] = 0
                        data = equal(data, 0.9)

                    # ----------- If Soft label ---------------#
                    # annotation[annotation == 255] = 1
                    annotation[annotation > 0] = 1
                    # annotation = annotation.astype(np.float32) / 255
                    # --------------------------------------#

                    # flg = int(round(random.random()))
                    if in_channel == 1:
                        data_final = data_ori
                    else:
                        data_final = np.concatenate((data_ori, data[np.newaxis, ...]), axis=0)

                    self.datas.append(data_final.astype(np.float32))
                    self.labels.append(annotation[np.newaxis, ...].astype(np.float32))

                    total_volumes += 1
                    pbar.update()

            if data_mix:
                artifacts_folder_path = data_path + '/artifacts/'
                artifacts_path = get_dir(artifacts_folder_path)
                with tqdm(total=len(volumes_path) * n_samples, desc=f'artifacts numbers') as pbar:
                    for apath in artifacts_path:
                        ak_seed = random.randint(0, 3)
                        artifact = read_tiff_stack(apath)
                        if artifact.shape[0] < opt.input_dim or artifact.shape[1] < opt.input_dim \
                                or artifact.shape[2] < opt.input_dim:
                            print(artifact.shape)
                            continue
                        for _ in range(max(1, int(len(volumes_path) / len(artifacts_path) * n_samples * 0.7))):
                            z = random.randint(0, artifact.shape[0] - input_dim)
                            x = random.randint(0, artifact.shape[1] - input_dim)
                            y = random.randint(0, artifact.shape[2] - input_dim)
                            artifact = artifact[z:z + input_dim, x:x + input_dim, y:y + input_dim]
                            artifact = np.rot90(np.swapaxes(artifact, 0, 2), k=ak_seed).swapaxes(2, 0)
                            # artifact = equal(artifact, 0.9)
                            artifact = (artifact - artifact.min()) / (artifact.max() - artifact.min())
                            data_ori = artifact.copy()[np.newaxis, :, :, :]
                            # data_ori = artifact[np.newaxis, :, :, :]
                            annotation[annotation > 0] = 0
                            # else:
                            #     mix_seed = random.random() * 0.2 + 0.4  # from 0.4 to 0.6
                            #     data_ori = self.datas[random.randint(0, total_volumes_axon - 1)]
                            #     data_ori = (data_ori * mix_seed + artifact * (1 - mix_seed))

                            if in_channel == 1:
                                data_final = data_ori
                            else:
                                data_final = np.concatenate((data_ori, data[np.newaxis, ...]), axis=0)

                            self.datas.append(data_final.astype(np.float32))
                            self.labels.append(annotation[np.newaxis, ...].astype(np.float32))

                            total_volumes += 1
                            pbar.update()

        if unlabel and self.flag:
            self.labeled_num = total_volumes
            nonlabel_path = get_dir(data_path + '/nonlabel')
            with tqdm(total=len(nonlabel_path) * n_samples, desc=f'volumes for semi-supervised learning') as pbar:
                for vpath in nonlabel_path:
                    volume = read_tiff_stack(vpath)
                    for _ in range(n_samples):
                        z = random.randint(0, label.shape[0] - input_dim)
                        x = random.randint(0, label.shape[1] - input_dim)
                        y = random.randint(0, label.shape[2] - input_dim)
                        volume_chunk = volume[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                        if random.randint(0, 1) == 0:
                            volume_chunk = contrast_augmentation(volume_chunk, label_chunk, rad=8, N=3)
                        k_seed = random.randint(0, 3)
                        flip_seed = random.randint(0, 1)
                        volume_rot = np.rot90(np.swapaxes(volume_chunk, 0, 2), k=k_seed).swapaxes(2, 0)

                        data, _ = callfunc[flip_seed]()

                        # data_ori = data.copy()[np.newaxis, :, :, :]
                        # data_ori = equal(data, 0.9)[np.newaxis, :, :, :]
                        if in_channel != 1:
                            im = equal(gauss_cal(data), 0.1)
                            data[im < 0.1] = 0
                            data = equal(data, 0.9)

                        data_ori = ((data - data.min()) / (data.max() - data.min()))[np.newaxis, :, :, :]

                        if in_channel == 1:
                            data_final = data_ori
                        else:
                            data_final = np.concatenate((data_ori, data[np.newaxis, ...]), axis=0)

                        self.datas.append(data_final.astype(np.float32))
                        # self.labels.append(annotation[np.newaxis, ...].astype(np.float32))
                        self.labels.append(np.zeros_like(data_final).astype(np.float32))
                        total_volumes += 1

                        pbar.update()

        # dataset = np.array(self.datas)
        # self.datas = list(equal(dataset, 0.9))
        # print('max: %f, min: %f' % (dataset.max(), dataset.min()))

    def __getitem__(self, index):
        image = self.datas[index]
        # if self.flag == 'train' and random.randint(0, 1) == 0:
        #     reference = self.datas[random.randint(0, len(self) - 1)]
        #     matched = match_histograms(image, reference)
        #     return matched.astype(np.float32), self.labels[index]
        # else:
        return image, self.labels[index]

    def __len__(self):
        return len(self.datas)