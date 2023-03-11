import os
import time
import torch
import numpy as np

from PIL import Image
from torch.utils import data
import json
from joblib import Parallel, delayed


def parallel_load(img_dir, img_list, img_size, verbose=0):
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert("L").resize(
            (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)


class AnomalyDetectionDataset(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train", extra_data=0, ar=0.):
        super(AnomalyDetectionDataset, self).__init__()
        assert mode in ["train", "test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        if mode == "train":
            train_normal = data_dict["train"]["0"]

            normal_l = data_dict["train"]["unlabeled"]["0"]
            abnormal_l = data_dict["train"]["unlabeled"]["1"]
            if extra_data > 0:
                abnormal_num = int(extra_data * ar)
                normal_num = extra_data - abnormal_num
            else:
                abnormal_num = 0
                normal_num = 0

            train_l = train_normal + normal_l[:normal_num] + abnormal_l[:abnormal_num]
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), train_l, img_size)
            self.labels += (len(train_normal) + normal_num) * [0] + abnormal_num * [1]
            self.img_id += [img_name.split('.')[0] for img_name in train_l]
            print("Loaded {} normal images, "
                  "{} (unlabeled) normal images, "
                  "{} (unlabeled) abnormal images. {:.3f}s".format(len(train_normal), normal_num, abnormal_num,
                                                                   time.time() - t0))

        else:  # test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]
        return img, label, img_id

    def __len__(self):
        return len(self.slices)


class SelfAnomalyDataset(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None):
        super(SelfAnomalyDataset, self).__init__()
        self.root = main_path
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x
        self.anomaly_transform = self.transform

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        t0 = time.time()
        train_normal = data_dict["train"]["0"]
        self.slices += parallel_load(os.path.join(self.root, "images"), train_normal, img_size)
        print("Loaded {} normal images. {:.3f}s".format(len(train_normal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)
        if np.random.rand() > 0.5:
            img, mask = self.generate_anomaly(img, index, core_percent=0.8)
            label = 1
        else:
            mask = torch.zeros_like(img).squeeze().long()
            label = 0
        return img, label, mask

    def __len__(self):
        return len(self.slices)

    def generate_anomaly(self, image, index, core_percent=0.8):
        dims = np.array(np.shape(image)[1:])  # H x W
        core = core_percent * dims  # width of core region
        offset = (1 - core_percent) * dims / 2  # offset to center core

        min_width = np.round(0.05 * dims[1])
        max_width = np.round(0.2 * dims[1])  # make sure it is less than offset

        center_dim1 = np.random.randint(offset[0], offset[0] + core[0])
        center_dim2 = np.random.randint(offset[1], offset[1] + core[1])
        patch_center = np.array([center_dim1, center_dim2])
        patch_width = np.random.randint(min_width, max_width)

        coor_min = patch_center - patch_width
        coor_max = patch_center + patch_width

        # clip coordinates to within image dims
        coor_min = np.clip(coor_min, 0, dims)
        coor_max = np.clip(coor_max, 0, dims)

        alpha = torch.rand(1)  #
        mask = torch.zeros_like(image).squeeze()
        mask[coor_min[0]:coor_max[0], coor_min[1]:coor_max[1]] = alpha
        mask_inv = 1 - mask

        # mix
        anomaly_source_index = np.random.randint(0, len(self.slices))
        while anomaly_source_index == index:
            anomaly_source_index = np.random.randint(0, len(self.slices))
        anomaly_source = self.slices[anomaly_source_index]
        anomaly_source = self.anomaly_transform(anomaly_source)
        image_synthesis = mask_inv * image + mask * anomaly_source

        return image_synthesis, (mask > 0).long()
