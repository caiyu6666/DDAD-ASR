import time

import pandas as pd
import torch
import numpy as np
import random
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from test import test_single_model
from models import EntropyLossEncap, FocalLoss, get_model, get_loader, load_ab, AverageMeter


def train_module_ab(cfgs, opt):
    out_dir = cfgs["Exp"]["out_dir"]

    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]
    extra_data = Data["extra_data"]
    ar = Data["ar"]

    Solver = cfgs["Solver"]
    bs = Solver["bs"]
    lr = Solver["lr"]
    weight_decay = Solver["weight_decay"]
    num_epoch = Solver["num_epoch"]

    if opt.mode == "a":
        train_loader = get_loader(dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1,
                                  extra_data=extra_data, ar=ar)
    else:
        train_loader = get_loader(dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1,
                                  extra_data=0, ar=0)
    test_loader = get_loader(dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1)

    model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)

    writer = SummaryWriter(os.path.join(out_dir, "log"))
    if network in ["AE", "AE-U"]:
        model = AE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt)
    elif network == "MemAE":
        model = MemAE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt)
    writer.close()
    print()

    model_path = os.path.join(out_dir, "{}".format(opt.mode))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i in range(20):
        # For the index to save the trained model for ensemble.
        model_name = os.path.join(model_path, "{}.pth".format(i))
        if not os.path.exists(model_name):  # save
            torch.save(model.state_dict(), model_name)
            break


def train_refine(cfgs, refine_in):
    out_dir = cfgs["Exp"]["out_dir"]

    Model = cfgs["Model"]
    network = Model["network"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]

    Solver = cfgs["RefineSolver"]
    bs = Solver["bs"]
    lr = Solver["lr"]
    weight_decay = Solver["weight_decay"]
    num_epoch = Solver["num_epoch"]
    grad_clip = Solver["grad_clip"]

    module_a, module_b = load_ab(cfgs)  # UDM and NDM

    refine_net = get_model(network="refine", in_channels=len(refine_in), out_channels=2)

    train_loader = get_loader(dataset=dataset, dtype="train", bs=bs, img_size=img_size, workers=1, self_sup=True)
    test_loader = get_loader(dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1)
    optimizer = torch.optim.Adam(refine_net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)

    writer = SummaryWriter(os.path.join(out_dir, "log_refine"))
    refine_net, results, best_auc, best_ap = refine_trainer(refine_net, module_a, module_b, train_loader, test_loader,
                                                            optimizer, num_epoch, writer, network, refine_in, grad_clip)
    writer.close()

    method_name = "refine_dual" if len(refine_in) == 2 else "refine_intra"

    csv_name = "{}_results.csv".format(method_name)
    results.to_csv(os.path.join(out_dir, csv_name), index=False)
    results_str = "{}   Final AUC:{:.3f}  Final AP:{:.3f}    Best AUC:{:.3f}  Best AP:{:.3f}".format(
        method_name, results["AUC"].iloc[-1], results["AP"].iloc[-1], best_auc, best_ap)
    print(results_str)

    txt_name = "{}_results.txt".format(method_name)
    with open(os.path.join(out_dir, txt_name), "w") as f:
        f.write(results_str)

    model_path = os.path.join(out_dir, "refine")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # model_name = os.path.join(model_path, "refine_dual.pth") if len(refine_in) == 2 \
    #     else os.path.join(model_path, "refine_intra.pth")
    model_name = os.path.join(model_path, "{}.pth".format(method_name))
    torch.save(refine_net.state_dict(), model_name)


def refine_trainer(refine_net, module_a, module_b, train_loader, test_loader, optimizer, num_epoch, writer, network,
                   refine_in, grad_clip):
    assert network in ["AE", "MemAE", "AE-U"]
    criterion = FocalLoss()

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)

    # batch_num = len(train_loader)
    auc_l, ap_l = [], []
    best_auc, best_ap = 0., 0.
    for e in range(1, num_epoch + 1):
        refine_net.train()
        losses = AverageMeter()
        t0 = time.time()
        for batch_idx, (image, _, anomaly_mask) in enumerate(train_loader):
            image, anomaly_mask = image.cuda(), anomaly_mask.cuda()
            # image: bs x 1 x H x W; label: bs x H x W
            # net_in = [image]
            modules_out = []
            unc = []
            if network == "AE":
                for model in module_a:
                    modules_out.append(model(image))
                for model in module_b:
                    modules_out.append(model(image))
            elif network == "MemAE":
                for model in module_a:
                    modules_out.append(model(image)["output"])
                for model in module_b:
                    modules_out.append(model(image)["output"])
            elif network == "AE-U":
                for model in module_a:
                    mean, logvar = model(image)
                    modules_out.append(mean)
                for model in module_b:
                    mean, logvar = model(image)
                    modules_out.append(mean)
                    unc.append(torch.exp(logvar))
                unc = torch.cat(unc, dim=1)  # bs x K x h x w
                unc = torch.mean(unc, dim=1, keepdim=True)  # bs x 1 x h x w
            else:
                raise Exception("Invalid network: {}".format(network))

            out_a = torch.cat(modules_out[:len(module_a)], dim=1)
            out_b = torch.cat(modules_out[len(module_a):], dim=1)
            mu_a = torch.mean(out_a, dim=1, keepdim=True)
            mu_b = torch.mean(out_b, dim=1, keepdim=True)

            inter_dis = torch.abs(mu_a - mu_b)
            intra_dis = torch.std(out_b, dim=1, keepdim=True)
            if network == "AE-U":
                inter_dis /= torch.sqrt(unc)
                intra_dis /= torch.sqrt(unc)

            net_in = []
            if "inter_dis" in refine_in:
                net_in.append(inter_dis)
            if "intra_dis" in refine_in:
                net_in.append(intra_dis)

            net_in = torch.cat(net_in, dim=1)
            segmentation = refine_net(net_in)

            segmentation = torch.softmax(segmentation, dim=1)

            loss = criterion(segmentation, anomaly_mask)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(refine_net.parameters(), grad_clip)

            optimizer.step()
            # if (batch_idx + 1) % 100 == 0:
            #     print("Epoch[{}][{:3d}/{}]  Loss:{:.4f}".format(e, batch_idx+1, batch_num, loss.item()))

            losses.update(loss.item(), image.size(0))

        test_gt = []
        test_gap, test_gmp = [], []
        refine_net.eval()
        with torch.no_grad():
            # for image, label, _ in tqdm(test_loader):
            for image, label, _ in test_loader:
                test_gt.append(label.item())

                image, label = image.cuda(), label.cuda()
                # net_in = [image]
                modules_out = []
                unc = []
                if network == "AE":
                    for model in module_a:
                        modules_out.append(model(image))
                    for model in module_b:
                        modules_out.append(model(image))
                elif network == "MemAE":
                    for model in module_a:
                        modules_out.append(model(image)["output"])
                    for model in module_b:
                        modules_out.append(model(image)["output"])
                elif network == "AE-U":
                    for model in module_a:
                        mean, logvar = model(image)
                        modules_out.append(mean)
                    for model in module_b:
                        mean, logvar = model(image)
                        modules_out.append(mean)
                        unc.append(torch.exp(logvar))
                    unc = torch.cat(unc, dim=1)  # bs x K x h x w
                    unc = torch.mean(unc, dim=1, keepdim=True)  # bs x 1 x h x w
                else:
                    raise Exception("Invalid network: {}".format(network))

                out_a = torch.cat(modules_out[:len(module_a)], dim=1)
                out_b = torch.cat(modules_out[len(module_a):], dim=1)
                mu_a = torch.mean(out_a, dim=1, keepdim=True)
                mu_b = torch.mean(out_b, dim=1, keepdim=True)

                inter_dis = torch.abs(mu_a - mu_b)
                intra_dis = torch.std(out_b, dim=1, keepdim=True)
                if network == "AE-U":
                    inter_dis /= torch.sqrt(unc)
                    intra_dis /= torch.sqrt(unc)

                net_in = []
                if "inter_dis" in refine_in:
                    net_in.append(inter_dis)
                if "intra_dis" in refine_in:
                    net_in.append(intra_dis)

                net_in = torch.cat(net_in, dim=1)
                out_mask = refine_net(net_in)

                out_mask_sm = torch.softmax(out_mask, dim=1)
                out_mask_cv = out_mask_sm[:, 1:, :, :]  # 1 x 1 x H x W

                test_gap.append(np.mean(out_mask_cv.cpu().detach().numpy()))
                # Global Average Pooling to get the image score

        auc_gap = roc_auc_score(test_gt, test_gap)
        ap_gap = average_precision_score(test_gt, test_gap)

        if auc_gap > best_auc:
            best_auc = auc_gap
        if ap_gap > best_ap:
            best_ap = ap_gap

        writer.add_scalar('loss', losses.avg, e)

        writer.add_scalar('auc_gap', auc_gap, e)
        writer.add_scalar('ap_gap', ap_gap, e)

        auc_l.append(auc_gap)
        ap_l.append(ap_gap)

        print("Epoch[{}/{}]\tTime:{:.1f}s\tLoss:{:.3f}\t"
              "AUC:{:.3f}\tAP:{:.3f}\tBest AUC:{:.3f}\tBest AP:{:.3f}".format(
            e, num_epoch, time.time() - t0, losses.avg, auc_gap, ap_gap, best_auc, best_ap))

    results = {"epoch": [i for i in range(num_epoch)], "AUC": auc_l, "AP": ap_l}
    results = pd.DataFrame(results)
    return refine_net, results, best_auc, best_ap


def AE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt):
    t0 = time.time()
    for e in range(1, num_epoch + 1):
        l1s, l2s = [], []
        model.train()
        for (x, _, _) in train_loader:
            x = x.cuda()
            x.requires_grad = False
            if cfgs["Model"]["network"] == "AE":
                out = model(x)
                rec_err = (out - x) ** 2
                loss = rec_err.mean()
                l1s.append(loss.item())
            else:  # AE-U
                mean, logvar = model(x)
                rec_err = (mean - x) ** 2
                loss1 = torch.mean(torch.exp(-logvar) * rec_err)
                loss2 = torch.mean(logvar)
                loss = loss1 + loss2
                l1s.append(rec_err.mean().item())
                l2s.append(loss2.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l1s = np.mean(l1s)
        l2s = np.mean(l2s) if len(l2s) > 0 else 0
        writer.add_scalar('rec_err', l1s, e)
        writer.add_scalar('logvars', l2s, e)

        if e % 25 == 0 or e == 1:
            t = time.time() - t0
            t0 = time.time()
            if opt.mode == "b":
                auc, ap = test_single_model(model=model, test_loader=test_loader, cfgs=cfgs)
                writer.add_scalar('AUC', auc, e)
                writer.add_scalar('AP', ap, e)
                print("Mode {}. Epoch[{:3d}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f}   "
                      "Rec_err:{:.4f}  logvars:{:.4f}".format(opt.mode, e, num_epoch, t, auc, ap, l1s, l2s))
            else:
                print("Mode {}. Epoch[{:3d}/{}]  Time:{:.2f}s  "
                      "Rec_err:{:.4f}  logvars:{:.4f}".format(opt.mode, e, num_epoch, t, l1s, l2s))

    return model


def MemAE_trainer(model, train_loader, test_loader, optimizer, num_epoch, writer, cfgs, opt):
    criterion_entropy = EntropyLossEncap()
    entropy_loss_weight = cfgs["Model"]["entropy_loss_weight"]
    t0 = time.time()
    for e in range(1, num_epoch + 1):
        l1s = []
        ent_ls = []
        model.train()
        for (x, _, _) in train_loader:
            x = x.cuda()
            x.requires_grad = False
            out = model(x)
            rec = out['output']
            att_w = out['att']

            rec_err = (rec - x) ** 2
            loss1 = rec_err.mean()
            entropy_loss = criterion_entropy(att_w)
            loss = loss1 + entropy_loss_weight * entropy_loss

            l1s.append(rec_err.mean().item())
            ent_ls.append(entropy_loss.mean().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        l1s = np.mean(l1s)
        ent_ls = np.mean(ent_ls)
        writer.add_scalar('rec_err', l1s, e)
        writer.add_scalar('entropy_loss', ent_ls, e)
        if e % 25 == 0 or e == 1:
            t = time.time() - t0
            t0 = time.time()

            if opt.mode == "b":
                auc, ap = test_single_model(model=model, test_loader=test_loader, cfgs=cfgs)
                writer.add_scalar('AUC', auc, e)
                writer.add_scalar('AP', ap, e)
                print("Mode {}. Epoch[{:3d}/{}]  Time:{:.2f}s  AUC:{:.3f}  AP:{:.3f}   "
                      "Rec_err:{:.4f}   Entropy_loss:{:.4f}".format(opt.mode, e, num_epoch, t, auc, ap, l1s, ent_ls))
            else:
                print("Mode {}. Epoch[{:3d}/{}]  Time:{:.2f}s  "
                      "Rec_err:{:.4f}   Entropy_loss:{:.4f}".format(opt.mode, e, num_epoch, t, l1s, ent_ls))

    return model
