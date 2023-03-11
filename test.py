import torch
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os
from models import get_model, get_loader, load_ab
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def chi2_distance(A, B):
    # compute the chi-squared distance
    chi = 0.5 * np.sum([((a - b) ** 2 + 1e-15) / (a + b + 1e-15)
                        for (a, b) in zip(A, B)])

    return chi


def anomaly_score_histogram(y_score, y_true, anomaly_score, out_dir, f_name):
    plt.rcParams.update({'font.size': 20})
    plt.cla()
    normal_score, _, _ = plt.hist(y_score[y_true == 0], bins=100, range=(0, 1), density=True, color='blue', alpha=0.5,
                                  label="Normal")
    abnormal_score, _, _ = plt.hist(y_score[y_true == 1], bins=100, range=(0, 1), density=True, color='red', alpha=0.5,
                                    label="Abnormal")

    chi2_dis = chi2_distance(normal_score, abnormal_score)
    plt.text(0.5, 6, "$\chi^2$-dis={:.2f}".format(chi2_dis))

    plt.xlabel(anomaly_score)
    plt.ylabel("Frequency")
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 2, 4, 6, 8, 10])
    plt.xlim(0, 1)
    plt.ylim(0, 11)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('{}/{}.pdf'.format(out_dir, f_name))
    plt.savefig('{}/{}.png'.format(out_dir, f_name))


def test_rec(cfgs):
    gpu = cfgs["Exp"]["gpu"]
    Model = cfgs["Model"]
    network = Model["network"]
    mp = Model["mp"]
    ls = Model["ls"]
    mem_dim = Model["mem_dim"]
    shrink_thres = Model["shrink_thres"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]

    out_dir = cfgs["Exp"]["out_dir"]

    test_loader = get_loader(dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1)

    module_b = []
    for state_dict in os.listdir(os.path.join(out_dir, "b")):
        model = get_model(network=network, mp=mp, ls=ls, img_size=img_size, mem_dim=mem_dim, shrink_thres=shrink_thres)
        model.load_state_dict(torch.load(os.path.join(out_dir, "b", state_dict),
                                         map_location=torch.device('cuda:{}'.format(gpu))))
        model.eval()
        module_b.append(model)

    print("=> Testing ... ")
    auc_l = []
    ap_l = []
    for model in module_b:
        auc, ap = test_single_model(model, test_loader, cfgs)
        auc_l.append(auc)
        ap_l.append(ap)
    print("Average results:")
    print("AUC: ", np.mean(auc_l), "±", np.std(auc_l))
    print("AP: ", np.mean(ap_l), "±", np.std(ap_l))
    print("\nResults of each model:")
    print("AUC:", auc_l)
    print("AP:", ap_l)


def test_single_model(model, test_loader, cfgs):
    model.eval()
    network = cfgs["Model"]["network"]
    with torch.no_grad():
        y_score, y_true = [], []
        for bid, (x, label, img_id) in enumerate(test_loader):
            x = x.cuda()
            if network == "AE-U":
                out, logvar = model(x)
                rec_err = (out - x) ** 2
                res = torch.exp(-logvar) * rec_err
            elif network == "AE":
                out = model(x)
                rec_err = (out - x) ** 2
                res = rec_err
            elif network == "MemAE":
                recon_res = model(x)
                rec = recon_res['output']
                res = (rec - x) ** 2

            res = res.mean(dim=(1, 2, 3))

            y_true.append(label.cpu())
            y_score.append(res.cpu().view(-1))

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc = metrics.roc_auc_score(y_true, y_score)
        ap = metrics.average_precision_score(y_true, y_score)
        return auc, ap


def evaluate(cfgs):
    Model = cfgs["Model"]
    network = Model["network"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]

    out_dir = cfgs["Exp"]["out_dir"]

    test_loader = get_loader(dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1)

    module_a, module_b = load_ab(cfgs)

    print("=> Loaded {} models in UDM".format(len(module_a)))
    print("=> Loaded {} models in NDM".format(len(module_b)))

    print("=> Evaluating ... ")
    with torch.no_grad():
        y_true = []
        rec_err_l, inter_dis_l, intra_dis_l = [], [], []

        for x, label, img_id in tqdm(test_loader):
            x = x.cuda()
            if network == "AE":
                a_rec = torch.cat([model(x).squeeze(0) for model in module_a])  # N x h x w
                b_rec = torch.cat([model(x).squeeze(0) for model in module_b])
            elif network == "AE-U":
                a_rec = torch.cat([model(x)[0].squeeze(0) for model in module_a])  # N x h x w
                b_rec, unc = [], []
                for model in module_b:
                    mean, logvar = model(x)
                    b_rec.append(mean.squeeze(0))
                    unc.append(torch.exp(logvar).squeeze(0))
                b_rec = torch.cat(b_rec)
                unc = torch.cat(unc)
            elif network == "MemAE":
                a_rec = torch.cat([model(x)["output"].squeeze(0) for model in module_a])  # N x h x w
                b_rec = torch.cat([model(x)["output"].squeeze(0) for model in module_b])
            else:
                raise Exception("Invalid Network")
            mu_a = torch.mean(a_rec, dim=0)  # h x w
            mu_b = torch.mean(b_rec, dim=0)  # h x w

            # Image-Level discrepancy
            if network == "AE-U":
                var = torch.mean(unc, dim=0)

                rec_err = (x - mu_b) ** 2 / var
                inter_dis = torch.sqrt((mu_a - mu_b) ** 2 / var)
                intra_dis = torch.sqrt(torch.var(b_rec, dim=0) / var)
            else:
                rec_err = (x - mu_b) ** 2
                inter_dis = torch.abs(mu_a - mu_b)
                intra_dis = torch.std(b_rec, dim=0)

            rec_err_l.append(rec_err.mean().cpu())
            inter_dis_l.append(inter_dis.mean().cpu())
            intra_dis_l.append(intra_dis.mean().cpu())

            y_true.append(label.cpu().item())

        rec_err_l = np.array(rec_err_l)
        inter_dis_l = np.array(inter_dis_l)
        intra_dis_l = np.array(intra_dis_l)

        y_true = np.array(y_true)

        rec_auc = metrics.roc_auc_score(y_true, rec_err_l)
        rec_ap = metrics.average_precision_score(y_true, rec_err_l)

        intra_auc = metrics.roc_auc_score(y_true, intra_dis_l)
        intra_ap = metrics.average_precision_score(y_true, intra_dis_l)

        inter_auc = metrics.roc_auc_score(y_true, inter_dis_l)
        inter_ap = metrics.average_precision_score(y_true, inter_dis_l)

        rec_str = 'Rec. (ensemble)  AUC:{:.3f}  AP:{:.3f}'.format(rec_auc, rec_ap)
        intra_str = 'DDAD-intra       AUC:{:.3f}  AP:{:.3f}'.format(intra_auc, intra_ap)
        inter_str = 'DDAD-inter       AUC:{:.3f}  AP:{:.3f}'.format(inter_auc, inter_ap)
        print(rec_str)
        print(intra_str)
        print(inter_str)
        with open(os.path.join(out_dir, "results.txt"), "w") as f:
            f.write(rec_str + "\n")
            f.write(intra_str + "\n")
            f.write(inter_str + "\n")
        print()

        # Visualization histogram
        intra_dis_l = (intra_dis_l - np.min(intra_dis_l)) / (np.max(intra_dis_l) - np.min(intra_dis_l))
        rec_err_l = (rec_err_l - np.min(rec_err_l)) / (np.max(rec_err_l) - np.min(rec_err_l))
        inter_dis_l = (inter_dis_l - np.min(inter_dis_l)) / (np.max(inter_dis_l) - np.min(inter_dis_l))

        anomaly_score_histogram(y_score=intra_dis_l, y_true=y_true, anomaly_score="Intra-discrepancy", out_dir=out_dir,
                                f_name="intra_hist")
        anomaly_score_histogram(y_score=inter_dis_l, y_true=y_true, anomaly_score="Inter-discrepancy", out_dir=out_dir,
                                f_name="inter_hist")
        anomaly_score_histogram(y_score=rec_err_l, y_true=y_true, anomaly_score="Reconstruction error", out_dir=out_dir,
                                f_name="rec_hist")


def evaluate_r(cfgs, refine_in):
    gpu = cfgs["Exp"]["gpu"]

    Model = cfgs["Model"]
    network = Model["network"]

    Data = cfgs["Data"]
    dataset = Data["dataset"]
    img_size = Data["img_size"]

    out_dir = cfgs["Exp"]["out_dir"]

    test_loader = get_loader(dataset=dataset, dtype="test", bs=1, img_size=img_size, workers=1)
    module_a, module_b = load_ab(cfgs)

    refine_net = get_model(network="refine", in_channels=len(refine_in), out_channels=2)
    if len(refine_in) == 2:
        refine_net.load_state_dict(torch.load(os.path.join(out_dir, "refine", "refine_dual.pth"),
                                              map_location=torch.device('cuda:{}'.format(gpu))))
    else:
        refine_net.load_state_dict(torch.load(os.path.join(out_dir, "refine", "refine_intra.pth"),
                                              map_location=torch.device('cuda:{}'.format(gpu))))
    refine_net.eval()

    with torch.no_grad():
        image_score, image_true = [], []
        # for idx, (image, label, img_id) in enumerate(tqdm(test_loader)):
        for idx, (image, label, img_id) in enumerate(test_loader):
            # img_id = img_id[0]
            image_true.append(label.item())

            image = image.cuda()

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

            # net_in = [inter_dis, intra_dis]
            # net_in = [intra_dis]

            net_in = torch.cat(net_in, dim=1)
            out_mask = refine_net(net_in)  # 1 x 2 x H x W
            out_mask = torch.softmax(out_mask, dim=1)
            out_mask_cv = out_mask[:, 1:, :, :]
            image_score.append(np.mean(out_mask_cv.cpu().detach().numpy()))

    # print("Rec: {} ± {}".format(np.mean(rec_err_l), np.std(rec_err_l)))
    # print("Intra: {} ± {}".format(np.mean(intra_l), np.std(intra_l)))
    # print("Inter: {} ± {}".format(np.mean(inter_l), np.std(inter_l)))

    image_true = np.array(image_true)
    image_score = np.array(image_score)
    image_score = (image_score - np.min(image_score)) / (np.max(image_score) - np.min(image_score))

    histogram_name = "R_dual" if len(refine_in) == 2 else "R_intra"
    x_label = "$R_{dual}$" if len(refine_in) == 2 else "$R_{intra}$"
    anomaly_score_histogram(y_score=image_score, y_true=image_true, anomaly_score=x_label, out_dir=out_dir,
                            f_name=histogram_name)

    auc = metrics.roc_auc_score(image_true, image_score)
    ap = metrics.average_precision_score(image_true, image_score)

    print("{} - AUC:{:.3f}  AP:{:.3f}".format(histogram_name, auc, ap))
