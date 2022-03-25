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


class CurAxonsDataset(data.Dataset):
    def __init__(self, opt):
        data_path = opt.dataroot
        n_samples = opt.n_samples
        input_dim = opt.input_dim
        data_mix = not opt.noartifacts
        self.flag = opt.isTrain

        self.datas = []
        self.trans_datas = []
        self.labels = []
        self.data_matrix = []

        data_path = join(data_path, 'train') if self.flag else join(data_path, 'val')
        self.data_path = data_path
        volumes_folder_path = join(data_path, "volumes")
        transvolumes_folder_path = join(data_path, "transvolumes")
        labels_folder_path = join(data_path, "labels_sk") if self.flag else join(data_path, 'labels')

        volumes_path = get_dir(volumes_folder_path)
        transvolumes_path = get_dir(transvolumes_folder_path)
        labels_path = get_dir(labels_folder_path)

        assert len(labels_path) == len(volumes_path)
        if n_samples == None:
            n_samples = len(labels_path)

        callfunc = {
            0: lambda: [np.fliplr(volume_rot), np.fliplr(tvolume_rot), np.fliplr(label_rot)],
            1: lambda: [np.flipud(volume_rot), np.flipud(tvolume_rot), np.flipud(label_rot)],
         }

        total_volumes = 0
        with tqdm(total=len(volumes_path) * n_samples, desc=f'volumes numbers') as pbar:
            for vpath, tpath, lpath in zip(volumes_path, transvolumes_path, labels_path):
                # assert (vpath.split('/')[-1].replace('volume', 'label')) == lpath.split('/')[-1]
                volume = read_tiff_stack(vpath)
                tvolume = read_tiff_stack(tpath)
                label = read_tiff_stack(lpath)
                if volume.shape[0] < opt.input_dim or volume.shape[1] < opt.input_dim \
                        or volume.shape[2] < opt.input_dim:
                    continue
                for _ in range(n_samples):
                    z = random.randint(0, label.shape[0] - input_dim)
                    x = random.randint(0, label.shape[1] - input_dim)
                    y = random.randint(0, label.shape[2] - input_dim)
                    volume_chunk = volume[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    tvolume_chunk = tvolume[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    label_chunk = label[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    if random.randint(0, 1) == 0:
                        volume_chunk = contrast_augmentation(volume_chunk, label_chunk, rad=15, N=3)

                    k_seed = random.randint(0, 3)
                    flip_seed = random.randint(0, 1)

                    volume_rot = np.rot90(np.swapaxes(volume_chunk, 0, 2), k=k_seed).swapaxes(2, 0)
                    tvolume_rot = np.rot90(np.swapaxes(tvolume_chunk, 0, 2), k=k_seed).swapaxes(2, 0)
                    label_rot = np.rot90(np.swapaxes(label_chunk, 0, 2), k=k_seed).swapaxes(2, 0)

                    data, tdata, annotation = callfunc[flip_seed]()
                    data_ori = ((data - data.min()) / (data.max() - data.min()))[np.newaxis, :, :, :]
                    tdata_ori = ((tdata - tdata.min()) / (tdata.max() - tdata.min()))[np.newaxis, :, :, :]
                    annotation[annotation > 0] = 1

                    self.datas.append(data_ori.astype(np.float32))
                    self.data_matrix.append(np.zeros_like(data_ori))
                    self.trans_datas.append(tdata_ori.astype(np.float32))
                    self.labels.append(annotation[np.newaxis, ...].astype(np.float32))

                    total_volumes += 1
                    pbar.update()
            total_volumes_axon = total_volumes
            if data_mix:
                artifacts_folder_path = data_path + '/artifacts/'
                artifacts_path = get_dir(artifacts_folder_path)
                with tqdm(total=max(len(volumes_path) * n_samples, len(artifacts_path)), desc=f'artifacts numbers') as pbar:
                    for apath in artifacts_path:
                        ak_seed = random.randint(0, 3)
                        artifact = read_tiff_stack(apath)
                        if artifact.shape[0] < opt.input_dim or artifact.shape[1] < opt.input_dim \
                                or artifact.shape[2] < opt.input_dim:
                            print(artifact.shape)
                            continue
                        for _ in range(max(1, int(len(volumes_path) / len(artifacts_path) * n_samples))):
                            z = random.randint(0, artifact.shape[0] - input_dim)
                            x = random.randint(0, artifact.shape[1] - input_dim)
                            y = random.randint(0, artifact.shape[2] - input_dim)
                            artifact = artifact[z:z + input_dim, x:x + input_dim, y:y + input_dim]
                            artifact = np.rot90(np.swapaxes(artifact, 0, 2), k=ak_seed).swapaxes(2, 0)
                            # artifact = equal(artifact, 0.9)
                            artifact = (artifact - artifact.min()) / (artifact.max() - artifact.min())
                            data_ori = artifact.copy()[np.newaxis, :, :, :]
                            if random.randint(0, 1) == 0:
                                mix_seed = random.random() * 0.2 + 0.4  # from 0.4 to 0.6
                                idx = random.randint(0, total_volumes_axon - 1)
                                data_axon = self.datas[idx]
                                data_ori = (data_ori * mix_seed + data_axon * (1 - mix_seed))
                                annotation = self.labels[idx][0]
                            else:
                                annotation[annotation > 0] = 0
                            self.datas.append(data_ori.astype(np.float32))
                            self.labels.append(annotation[np.newaxis, ...].astype(np.float32))

                            total_volumes += 1
                            pbar.update()

    def update(self, epsilon, data_sign, index):
        for d, i in zip(data_sign, index):
            if i < len(self.data_matrix):
                i = i.cpu().int()
                self.data_matrix[i] += (d * epsilon).cpu().numpy()

    def __getitem__(self, index):
        if index < len(self.data_matrix):
            image = self.datas[index] * (1 - self.data_matrix[index]) + self.data_matrix[index] * self.trans_datas[index]
            return image, self.labels[index], index
        else:
            return self.datas[index], self.labels[index], index

    def __len__(self):
        return len(self.datas)
