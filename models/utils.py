import torch
from models import AE, MemAE, RefineNetwork
from anomaly_data import AnomalyDetectionDataset, SelfAnomalyDataset
from torchvision import transforms
from torch.utils import data
import os


def get_model(network, in_channels=None, out_channels=None, mp=None, ls=None, img_size=None, mem_dim=None,
              shrink_thres=0.0):
    if network == "AE":
        model = AE(latent_size=ls, multiplier=mp, unc=False, img_size=img_size)
    elif network == "AE-U":
        model = AE(latent_size=ls, multiplier=mp, unc=True, img_size=img_size)
    elif network == "MemAE":
        model = MemAE(latent_size=ls, multiplier=mp, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
    elif network == "refine":
        model = RefineNetwork(in_channels=in_channels, out_channels=out_channels)
    else:
        raise Exception("Invalid Model Name!")

    model.cuda()
    return model


def get_loader(dataset, dtype, bs, img_size, workers=1, extra_data=0, ar=0., self_sup=False):
    DATA_PATH = os.path.join(os.path.expanduser("~"), "Med-AD")

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print("Dataset: {}".format(dataset))
    if dataset == 'rsna':
        path = os.path.join(DATA_PATH, 'RSNA')
    elif dataset == 'vin':
        path = os.path.join(DATA_PATH, "VinCXR")
    elif dataset == 'brain':
        path = os.path.join(DATA_PATH, "BrainTumor")
    elif dataset == 'lag':
        path = os.path.join(DATA_PATH, "LAG")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))

    if self_sup:
        dset = SelfAnomalyDataset(main_path=path, img_size=img_size, transform=transform)
    else:
        dset = AnomalyDetectionDataset(main_path=path, transform=transform, mode=dtype, img_size=img_size,
                                       extra_data=extra_data, ar=ar)

    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=True)

    return dataloader


def load_ab(cfgs, requires_grad=False):
    gpu = cfgs["Exp"]["gpu"]
    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]

    Data = cfgs["Data"]
    img_size = Data["img_size"]

    out_dir = cfgs["Exp"]["out_dir"]

    module_a = []
    for state_dict in sorted(os.listdir(os.path.join(out_dir, "a")), key=lambda x: int(x.split(".")[0])):
        model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
        model.load_state_dict(torch.load(os.path.join(out_dir, "a", state_dict),
                                         map_location=torch.device('cuda:{}'.format(gpu))))
        model.eval()
        if not requires_grad:
            for param in model.parameters():
                param.requires_grad = False
        module_a.append(model)

    module_b = []
    for state_dict in sorted(os.listdir(os.path.join(out_dir, "b")), key=lambda x: int(x.split(".")[0])):
        model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
        model.load_state_dict(torch.load(os.path.join(out_dir, "b", state_dict),
                                         map_location=torch.device('cuda:{}'.format(gpu))))
        model.eval()
        if not requires_grad:
            for param in model.parameters():
                param.requires_grad = False
        module_b.append(model)

    return module_a, module_b


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

