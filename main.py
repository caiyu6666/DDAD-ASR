import yaml
from trainer import *
from test import evaluate, test_rec, evaluate_r
from argparse import ArgumentParser

torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config', type=str, default="config/RSNA_AE.yaml")  # config file
    parser.add_argument('--mode', dest='mode', type=str, default=None, help="e.g., a, b, r, eval, eval_r, test")
    parser.add_argument('--refine', dest='refine_in', type=str, default="dual", help="dual, intra")

    """
    Description of modes.
    a: train one network in the Unknown Distribution Module (UDM) on normal+unlabeled data
    b: train one network in the Normative Distribution Module (NDM) on normal data
    r: train the ASR-Net (after finishing the training of UDM and NDM).
    eval: evaluate the reconstruction ensemble and DDAD (without ASR-Net).
    eval_r: evaluate the DDAD with ASR-Net
    test: test the reconstruction baselines
    """

    opt = parser.parse_args()

    with open(opt.config, "r") as f:
        cfgs = yaml.safe_load(f)

    torch.cuda.set_device(cfgs["Exp"]["gpu"])

    out_dir = cfgs["Exp"]["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if opt.refine_in == "dual":
        refine_in = ["inter_dis", "intra_dis"]  # For R_{dual}
    elif opt.refine_in == "intra":
        refine_in = ["intra_dis"]  # For R_{intra}
    else:
        raise Exception("Invalid refine in: {}".format(opt.refine_in))

    if opt.mode == "a" or opt.mode == "b":
        module = "UDM" if opt.mode == "a" else "NDM"
        print("=> Training the {}".format(module))
        train_module_ab(cfgs, opt)
    elif opt.mode == "r":
        print("=> Training the ASR-Net ...")
        if len(refine_in) == 2:
            print("=> Refine dual discrepancies")
        else:
            print("=> Refine intra-discrepancy")
        train_refine(cfgs, refine_in)
    elif opt.mode == "eval":
        print("=> Evaluating DDAD without ASR ...")
        evaluate(cfgs)
    elif opt.mode == "eval_r":
        print("=> Evaluating DDAD with ASR ...")
        evaluate_r(cfgs, refine_in)
    elif opt.mode == "test":
        print("=> Testing the reconstruction models ...")
        test_rec(cfgs)
    else:
        raise Exception("Invalid mode: {}".format(opt.mode))
