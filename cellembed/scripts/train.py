import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.optim as optim
import argparse
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use('Agg')

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, help="mode")
parser.add_argument("--host", type=str, help="host")
parser.add_argument("--port", type=int, help="port")

# basic usage
parser.add_argument("-d_p", "--data_path", type=str, default=r"../datasets", help="Path to the .pth file")
parser.add_argument("-data", "--dataset", type=str, default="All_train", help="Name of the dataset to load")

parser.add_argument('-source', '--source_dataset', default="all", type=str,
                    help="Which datasets to use for training. Input is 'all' or a list of datasets (e.g. [TNBC_2018,LyNSeC,IHC_TMA,CoNSeP])")
parser.add_argument("-m_f", "--model_folder", type=str, default=None,
                    help="Name of the model to resume training. This must be a folder inside model_path")
parser.add_argument("-m_p", "--model_path", type=str, default=r"../models",
                    help="Path to the folder containing the models")
parser.add_argument("-o_p", "--output_path", type=str, default=r"../All_train",
                    help="Path to the folder where the results will be saved")

parser.add_argument("-optim", "--optimizer", type=str, default="adamw", help="Optimizer to use, adam, sgd or adamw")
parser.add_argument("-m", "--model_str", type=str, default="cellembed", help="Model backbone to use")
parser.add_argument("-e_s", "--experiment_str", type=str, default="CEmbed SAM", help="String to identify the experiment")
parser.add_argument('-target', '--target_segmentation', default="C", type=str,
                    help=" Cells or nuclei or both? Accepts: C,N, NC")

parser.add_argument('-multihead', '--multihead', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to branch the decoder into multiple heads.")
parser.add_argument("-n_sigma", "--n_sigma", type=int, default=2, help="Number of sigma channels, must be at least 1")
parser.add_argument('-dim_coords', '--dim_coords', default=2, type=int,
                    help="Dimensionality of the coordinate system. Little support for anything but 2")

parser.add_argument('-mlp_w', '--mlp_width', default=10, type=int, help="Width of the MLP hidden dim")

parser.add_argument("-d", "--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('-num_workers', '--num_workers', default=4, type=int,
                    help="Number of CPU cores to use for data loading")
parser.add_argument('-pixel_size', '--requested_pixel_size', default=None, type=float,
                    help="Requested pixel size to rescale the input images")

# advanced usage
parser.add_argument("-bs", "--batch_size", type=int, default=1)
parser.add_argument("-e", "--num_epochs", type=int, default=200)
parser.add_argument('-len_epoch', '--length_of_epoch', default=10, type=int, help="Number of samples per epoch")
parser.add_argument("-lr", "--lr", type=float, default=0.001, help="Learning rate")

parser.add_argument("-s", "--save", type=bool, default=True,
                    help="Whether to save model outputs every time a new best F1 score is achieved")
parser.add_argument("-cluster", "--on_cluster", type=bool, default=False,
                    help="Flag to disable tqdm progress bars and other non-essential outputs, useful for running on a cluster")
parser.add_argument("-w", "--weight", default=True, type=lambda x: (str(x).lower() == 'true'),
                    help="Weight the random sampler in the training set to oversample images with more instances")
parser.add_argument("-layers", "--layers", type=str, default="[32, 64, 128, 256]", help="UNet layers")
parser.add_argument("-slice", "--data_slice", type=int, default=None,
                    help="Slice of the dataset to use, useful for debugging (e.g. only train on 1 image)")

parser.add_argument("-clip", "--clip", type=float, default=20, help="Gradient clipping value")
parser.add_argument("-decay", "--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("-drop", "--dropprob", type=float, default=0.,
                    help="Dropout probability, Not implemented for InstanSeg_UNet")
parser.add_argument("-tf", "--transform_intensity", type=float, default=0.5, help="Intensity transformation factor")
parser.add_argument("-dim_in", "--dim_in", type=int, default=3,
                    help="Number of channels that the (backbone) model expects. This is also the number of channels a channel invariant model would output.")

parser.add_argument("-dummy", "--dummy", default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Use the training set as a validation set, this will trigger a warning message. use only for debugging")
parser.add_argument('-to_centre', '--to_centre', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to use the instance centroid or the learnt instance centroid in InstanSeg")
parser.add_argument('-bg_weight', '--bg_weight', default=None, type=float,
                    help="Weight to assign to the background class in the loss function")
parser.add_argument('-opl', '--only_positive_labels', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Sample local maxima from the whole image, adds an object classifier")
parser.add_argument('-multi_centre', '--multi_centre', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help="Allow multi centres per instance, uses local maxima algorithm")
parser.add_argument('-open_license', '--open_license', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to filter out images that do not have an open license during training")
parser.add_argument('-modality', '--image_modality', default="all", type=str,
                    help="Filter out images that do not have this modality: Brightfield, Fluorescence, all")

parser.add_argument('-binary_loss_fn', '--binary_loss_fn', default="lovasz_hinge", type=str,
                    help="Loss function to use for instance segmentation: lovasz_hinge or dice_loss are supported. lovasz_hinge is a lot slower to start converging")
parser.add_argument('-seed_loss_fn', '--seed_loss_fn', default="l1_distance", type=str,
                    help="Loss function to use for seed selection, only binary_xloss and l1_distance are supported. Binary_xloss is much faster, but l1_distance is usually more accurate")

parser.add_argument('-o_h', '--optimize_hyperparameters', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to optimize hyperparameters every 10 epochs")
parser.add_argument('-window', '--window_size', default=128, type=int,
                    help="Size of the window containing each instance")
parser.add_argument('-norm', '--norm', default="BATCH", type=str,
                    help="Norm layer to use: None, INSTANCE, INSTANCE_INVARIANT, BATCH")
parser.add_argument('-augmentation_type', '--augmentation_type', default="minimal", type=str,
                    help="'minimal' or 'heavy' or 'brightfield_only'")
parser.add_argument('-adaptor_net', '--adaptor_net_str', default="1", type=str, help="Adaptor net to use")
parser.add_argument('-freeze', '--freeze_main_model', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to freeze the main model")
parser.add_argument('-f_e', '--feature_engineering', default="0", type=str, help="Feature engineering function to use")
parser.add_argument("-f", "--f", default=None, type=str, help="ignore, this is for jypyter notebook compatibility")

parser.add_argument('-use_deterministic', '--use_deterministic', default=False,
                    type=lambda x: (str(x).lower() == 'true'),
                    help="Whether to use deterministic algorithms (default=False)")
parser.add_argument('-tile', '--tile_size', default=256, type=int, help="Tile sizes for the input images")

parser.add_argument('-rng_seed', '--rng_seed', default=42, type=int,
                    help="Optional seed for the random number generator")
parser.add_argument('-samples', '--samples', default=True, type=lambda x: (str(x).lower() == 'true'), help="Samples")
parser.add_argument('-encoder', '--sam_encoder', default="large", type=str, help="'base' or 'large' or 'huge'")
parser.add_argument('-num_nuclei_classes', '--num_nuclei_classes', default=1, type=int, help="num_nuclei_classes")


def main(model, loss_fn, train_loader, test_loader, num_epochs=1000, epoch_name='output_epoch'):
    from cellembed.utils.AI_utils import optimize_hyperparameters, train_epoch_cellembed, test_epoch_cellembed, \
        train_epoch_cellposesam, test_epoch_cellposesam, train_epoch_microsam, test_epoch_microsam, \
        train_epoch_cellvit, test_epoch_cellvit, train_epoch_mediar, test_epoch_mediar, \
        train_epoch, test_epoch

    global best_f1_score, device, method_cellembed, method_instanseg, iou_threshold, args, optimizer, scheduler

    train_losses = []
    test_losses = []

    seed_loss = None
    label_loss = None

    seed_losses = []
    label_losses = []

    best_f1_score = -1
    f1_list = []
    f1_list_cells = []

    for epoch in range(num_epochs):

        print("Epoch:", epoch)

        if args.model_str.lower() == "cellembed":
            train_loss, train_time = train_epoch_cellembed(model, device, train_loader, loss_fn, optimizer, args=args)
        elif args.model_str.lower() == "instanseg_unet":
            train_loss, train_time = train_epoch(model, device, train_loader, loss_fn, optimizer, args=args)
        elif args.model_str.lower() == "cellposesam" or args.model_str.lower() == "cellpose3":
            train_loss, train_time = train_epoch_cellposesam(model, device, train_loader, optimizer, args=args)
        elif args.model_str.lower() == "microsam":
            train_loss, train_time = train_epoch_microsam(model, device, train_loader, optimizer, args=args)
        elif args.model_str.lower() == "cellvit":
            train_loss, train_time = train_epoch_cellvit(model, device, train_loader, optimizer, args=args)
        elif args.model_str.lower() == "mediar":
            train_loss, train_time = train_epoch_mediar(model, device, train_loader, optimizer, args=args)

        if epoch <= 5 and not args.model_folder:  # Training is just starting AND we are not loading a model
            save_epoch_outputs = True
        else:
            save_epoch_outputs = False

        if args.model_str.lower() == "cellembed":
            test_loss, seed_loss, label_loss, f1_score, test_time, f1_05 = test_epoch_cellembed(model, device,
                                                                                                test_loader,
                                                                                                loss_fn, debug=False,
                                                                                                best_f1=best_f1_score,
                                                                                                save_bool=save_epoch_outputs,
                                                                                                args=args,
                                                                                                postprocessing_fn=method_cellembed.postprocessing_seed,
                                                                                                method=method_cellembed,
                                                                                                iou_threshold=iou_threshold,
                                                                                                save_str=str(
                                                                                                    args.output_path / str(
                                                                                                        f"epoch_outputs/{epoch_name}_" + str(
                                                                                                            epoch))))

            if scheduler is not None:
                scheduler.step()
                print(f"Current pixel learning rate: {optimizer.param_groups[0]['lr']}")

            seed_losses.append(seed_loss)
            label_losses.append(label_loss)

        elif args.model_str.lower() == "instanseg_unet":

            test_loss, f1_score, test_time, f1_05 = test_epoch(model, device, test_loader,
                                                                                      loss_fn, debug=False,
                                                                                      best_f1=best_f1_score,
                                                                                      save_bool=save_epoch_outputs,
                                                                                      args=args,
                                                                                      postprocessing_fn=method_instanseg.postprocessing,
                                                                                      method=method_instanseg,
                                                                                      iou_threshold=iou_threshold,
                                                                                      save_str=str(
                                                                                          args.output_path / str(
                                                                                              f"epoch_outputs/{epoch_name}_" + str(
                                                                                                  epoch))))
            if scheduler is not None:
                scheduler.step()
                print(f"Current pixel learning rate: {optimizer.param_groups[0]['lr']}")

        elif args.model_str.lower() == "cellposesam" or args.model_str.lower() == "cellpose3":
            test_loss, f1_score, test_time, f1_05 = test_epoch_cellposesam(model, device, test_loader,
                                                                           debug=False,
                                                                           best_f1=best_f1_score,
                                                                           save_bool=save_epoch_outputs,
                                                                           args=args,
                                                                           iou_threshold=iou_threshold,
                                                                           save_str=str(
                                                                               args.output_path / str(
                                                                                   f"epoch_outputs/{epoch_name}_" + str(
                                                                                       epoch))))

            if scheduler is not None:
                scheduler.step()
                print(f"Current pixel learning rate: {optimizer.param_groups[0]['lr']}")

        elif args.model_str.lower() == "microsam":
            test_loss, f1_score, test_time, f1_05 = test_epoch_microsam(model, device, test_loader,
                                                                        debug=False,
                                                                        best_f1=best_f1_score,
                                                                        save_bool=save_epoch_outputs,
                                                                        args=args,
                                                                        iou_threshold=iou_threshold,
                                                                        save_str=str(
                                                                            args.output_path / str(
                                                                                f"epoch_outputs/{epoch_name}_" + str(
                                                                                    epoch))))

            scheduler.step(f1_score)
            print(f"Epoch {epoch} | lr: {optimizer.param_groups[0]['lr']}")

        elif args.model_str.lower() == "cellvit":
            test_loss, f1_score, test_time, f1_05 = test_epoch_cellvit(model, device, test_loader,
                                                                       debug=False,
                                                                       best_f1=best_f1_score,
                                                                       save_bool=save_epoch_outputs,
                                                                       args=args,
                                                                       iou_threshold=iou_threshold,
                                                                       save_str=str(
                                                                           args.output_path / str(
                                                                               f"epoch_outputs/{epoch_name}_" + str(
                                                                                   epoch))))

            if scheduler is not None:
                scheduler.step()
                print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        elif args.model_str.lower() == "mediar":

            test_loss, f1_score, test_time, f1_05 = test_epoch_mediar(model, device, test_loader,
                                                                      debug=False,
                                                                      best_f1=best_f1_score,
                                                                      save_bool=save_epoch_outputs,
                                                                      args=args,
                                                                      iou_threshold=iou_threshold,
                                                                      save_str=str(
                                                                          args.output_path / str(
                                                                              f"epoch_outputs/{epoch_name}_" + str(
                                                                                  epoch))))
            if scheduler is not None:
                scheduler.step()
                print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if seed_loss is not None:
            dict_to_print = {"train_loss": train_loss, "test_loss": test_loss, "seed_loss": seed_loss,
                             "label_loss": label_loss, "training_time": int(train_time),
                             "testing_time": int(test_time)}
        else:
            dict_to_print = {"train_loss": train_loss, "test_loss": test_loss, "training_time": int(train_time),
                             "testing_time": int(test_time)}

        if args.cells_and_nuclei:
            f1_list.append(f1_score[0])
            f1_list_cells.append(f1_score[1])
            dict_to_print["f1_score_nuclei"] = f1_score[0]
            dict_to_print["f1_score_cells"] = f1_score[1]

            dict_to_print["f1_05_nuclei"] = f1_05[0]
            dict_to_print["f1_05_cells"] = f1_05[1]

            f1_score = np.nanmean(f1_score)
            dict_to_print["f1_score_joint"] = f1_score

            f1_05 = np.nanmean(f1_05)
            dict_to_print["f1_05_joint"] = f1_05

        else:
            f1_score = f1_score[0]
            f1_05 = f1_05[0]
            f1_list.append(f1_score)
            dict_to_print["f1_score"] = f1_score
            dict_to_print["f1_05"] = f1_05

        if f1_score > best_f1_score or save_epoch_outputs:
            best_f1_score = np.maximum(f1_score, best_f1_score)

            print("Saving model, best f1_score:", best_f1_score)

            torch.save({
                'f1_score': float(best_f1_score),
                'epoch': int(epoch),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.output_path / "model_weights.pth")

        print(str(dict_to_print).replace("{", "").replace("}", "").replace("'", ""))

    return model, train_losses, test_losses, seed_losses, label_losses, f1_list, f1_list_cells


from typing import Dict


def training(segmentation_dataset: Dict = None, **kwargs):
    global device, method_cellembed, method_instanseg, iou_threshold, args, optimizer, scheduler
    args = parser.parse_args()

    from cellembed.utils.utils import plot_average, _choose_device
    from cellembed.utils.model_loader import build_model_from_dict, load_model_weights
    from cellembed.utils.data_loader import _read_images_from_pth, get_loaders

    args.data_path = Path(args.data_path)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    args.output_path = Path(args.output_path) / args.experiment_str
    print("Saving results to {}".format(os.path.abspath(args.output_path)))
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    os.environ["INSTANSEG_DATASET_PATH"] = os.environ.get("INSTANSEG_DATASET_PATH", str(args.data_path))
    os.environ["INSTANSEG_OUTPUT_PATH"] = os.environ.get("INSTANSEG_OUTPUT_PATH", str(args.output_path))

    # Seed as many rngs as we can
    if args.rng_seed:
        print(f'Setting RNG seed to {args.rng_seed}')
        torch.manual_seed(args.rng_seed)
        torch.cuda.manual_seed_all(args.rng_seed)
        np.random.seed(args.rng_seed)
        import random
        random.seed(args.rng_seed)
    else:
        print('RNG seed not set')

    if args.use_deterministic:
        print('Setting use_deterministic_algorithms=True')
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    args.layers = eval(args.layers)
    args_dict = vars(args)
    num_epochs = args.num_epochs
    n_sigma = args.n_sigma
    on_cluster = args.on_cluster
    dim_in = args.dim_in

    if args.norm == "None":
        args.norm = None

    if len(args.target_segmentation) == 2:
        args.cells_and_nuclei = True
    else:
        args.cells_and_nuclei = False

    device = _choose_device(args.device)

    if args.model_str.lower() == "instanseg_unet":

        from cellembed.utils.loss.instanseg_loss import InstanSeg
        method_instanseg = InstanSeg(binary_loss_fn_str=args.binary_loss_fn,
                                     seed_loss_fn=args.seed_loss_fn,
                                     device=device,
                                     n_sigma=n_sigma,
                                     cells_and_nuclei=args.cells_and_nuclei,
                                     to_centre=args.to_centre,
                                     window_size=args.window_size,
                                     dim_coords=args.dim_coords,
                                     multi_centre=args.multi_centre,
                                     feature_engineering_function=args.feature_engineering,
                                     bg_weight=args.bg_weight)

        def loss_fn(*args, **kwargs):
            return method_instanseg.forward(*args, **kwargs)

        dim_out = method_instanseg.dim_out
        args.dim_out = dim_out

    elif args.model_str.lower() == "cellembed":

        from cellembed.utils.loss.cellembed_loss import InstanSeg
        method_cellembed = InstanSeg(binary_loss_fn_str=args.binary_loss_fn,
                                     seed_loss_fn=args.seed_loss_fn,
                                     device=device,
                                     n_sigma=n_sigma,
                                     cells_and_nuclei=args.cells_and_nuclei,
                                     to_centre=args.to_centre,
                                     window_size=args.window_size,
                                     dim_coords=args.dim_coords,
                                     multi_centre=args.multi_centre,
                                     feature_engineering_function=args.feature_engineering,
                                     num_nuclei_classes=args.num_nuclei_classes)

        def loss_fn(*args, **kwargs):
            return method_cellembed.forward(*args, **kwargs)

        dim_out = method_cellembed.dim_out
        args.dim_out = dim_out

    else:

        loss_fn = None

    args_dict = vars(args)

    if int(dim_in) == 0:
        args_dict["dim_in"] = None
    else:
        args_dict["dim_in"] = int(dim_in)
    args_dict["dropprob"] = float(args.dropprob)

    model = build_model_from_dict(args_dict, random_seed=args.rng_seed)

    def get_optimizer(parameters, args):
        if args.optimizer.lower() == "adam":
            return optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "sgd":
            return optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == "adamw":
            return optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay, betas=(0.85, 0.85))
        else:
            raise NotImplementedError("Optimizer not recognized", args.optimizer)

    if args.model_str.lower() == "instanseg_unet":
        from cellembed.utils.loss.instanseg_loss import has_pixel_classifier_model
        if not has_pixel_classifier_model(model):
            model = method_instanseg.initialize_pixel_classifier(model, MLP_width=args.mlp_width)

    elif args.model_str.lower() == "cellembed":
        from cellembed.utils.loss.cellembed_loss import has_pixel_classifier_model
        if not has_pixel_classifier_model(model):
            model = method_cellembed.initialize_pixel_classifier(model, MLP_width=args.mlp_width)

    if args.model_folder:
        if args.model_folder == "None":
            args.model_folder = ""

        model, model_dict = load_model_weights(model, path=args.model_path, folder=args.model_folder, device=device,
                                               dict=args_dict)
        print("Resuming training from epoch", model_dict['epoch'])

    if "[" in args.source_dataset:
        args.source_dataset = args.source_dataset.replace("[", "").replace("]", "").replace("'", "").split(",")
    else:
        args.source_dataset = args.source_dataset

    scheduler = None

    # scheduler
    if args.model_str.lower() == "cellembed":

        optimizer = get_optimizer(model.parameters(), args)

        from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR
        warmup_epochs = 10
        first_decay_epoch = args.num_epochs - 150
        second_decay_epoch = args.num_epochs - 50

        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs)
        relative_first_decay = first_decay_epoch - warmup_epochs
        relative_second_decay = second_decay_epoch - warmup_epochs

        milestones_for_decay = [relative_first_decay, relative_second_decay]
        decay_scheduler = MultiStepLR(optimizer, milestones=milestones_for_decay, gamma=0.1)

        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_scheduler, decay_scheduler],
                                 milestones=[warmup_epochs])

    elif args.model_str.lower() == "instanseg_unet":

        args.optimtzer = "adam"
        optimizer = get_optimizer(model.parameters(), args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

    elif args.model_str.lower() == "cellposesam":

        args.lr = 0.00005
        args.optimtzer = "adamw"
        optimizer = get_optimizer(model.parameters(), args)

        from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR
        warmup_epochs = 10
        first_decay_epoch = args.num_epochs - 100
        second_decay_epoch = args.num_epochs - 50

        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_epochs)
        relative_first_decay = first_decay_epoch - warmup_epochs
        relative_second_decay = second_decay_epoch - warmup_epochs

        milestones_for_decay = [relative_first_decay, relative_second_decay]
        decay_scheduler = MultiStepLR(optimizer, milestones=milestones_for_decay, gamma=0.1)

        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_scheduler, decay_scheduler],
                                 milestones=[warmup_epochs])

    elif args.model_str.lower() == "cellpose3":

        args.lr = 0.001
        args.optimtzer = "adamw"
        optimizer = get_optimizer(model.parameters(), args)

        from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
        total_epochs = args.num_epochs
        warmup_epochs = 10
        decay_start_epoch = 300
        decay_factor = 0.5
        decay_interval = 10

        decay_milestones = []
        for i in range(decay_start_epoch, total_epochs, decay_interval):
            decay_milestones.append(i)

        if decay_milestones and decay_milestones[-1] >= total_epochs:
            decay_milestones.pop()

        warmup_scheduler = LinearLR(optimizer,
                                    start_factor=1e-8,
                                    end_factor=1.0,
                                    total_iters=warmup_epochs)

        relative_decay_milestones = [m - warmup_epochs for m in decay_milestones if m >= warmup_epochs]
        relative_decay_milestones = sorted(list(set([m for m in relative_decay_milestones if m >= 0])))
        decay_scheduler = MultiStepLR(optimizer,
                                      milestones=relative_decay_milestones,
                                      gamma=decay_factor)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_scheduler, decay_scheduler],
                                 milestones=[warmup_epochs])


    elif args.model_str.lower() == "microsam":

        args.lr = 0.0001
        args.optimtzer = "adam"
        optimizer = get_optimizer(model.parameters(), args)

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.9,
            patience=10
        )

    elif args.model_str.lower() == "cellvit":

        args.lr = 0.0003
        args.optimtzer = "adamw"
        optimizer = get_optimizer(model.parameters(), args)

        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=10, gamma=0.85)

    elif args.model_str.lower() == "mediar":

        args.lr = 0.00005
        args.optimtzer = "adamw"
        optimizer = get_optimizer(model.parameters(), args)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

    train_images, train_labels, train_meta, val_images, val_labels, val_meta = _read_images_from_pth(
        data_path=args.data_path,
        dataset=args.dataset,
        data_slice=args.data_slice,
        dummy=args.dummy,
        args=args,
        sets=["Train", "Validation"],
        complete_dataset=segmentation_dataset)

    train_loader, test_loader = get_loaders(train_images, train_labels, val_images, val_labels, train_meta, val_meta,
                                            args)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)

    if args.save:
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        if not os.path.exists(args.output_path / "epoch_outputs"):
            os.mkdir(args.output_path / "epoch_outputs")

    pd.DataFrame.from_dict(args_dict, orient='index').to_csv(args.output_path / "experiment_log.csv",
                                                             header=False)

    iou_threshold = np.linspace(0.5, 1.0, 10)

    model, train_losses, test_losses, seed_losses, label_losses, f1_list, f1_list_cells = main(model, loss_fn,
                                                                                               train_loader,
                                                                                               test_loader,
                                                                                               num_epochs=num_epochs)

    # from cellembed.utils.model_loader import load_model
    # model, model_dict = load_model(folder="", path=args.output_path)
    # model.eval()
    # model.to(device)

    df = pd.DataFrame({"train_loss": train_losses, "test_loss": test_losses, "f1_score": f1_list})
    df.to_csv(args.output_path / "experiment_metrics.csv", index=False, header=True)

    fig = plot_average(train_losses, test_losses, window_size=len(train_losses) // 10 + 1)
    plt.savefig(args.output_path / "loss.png")
    plt.close()

    if args.cells_and_nuclei:
        fig = plt.plot(f1_list, label="f1 score nuclei")
        plt.plot(f1_list_cells, label="f1 score cells")
        plt.ylim(0, 1)
        plt.savefig(args.output_path / "f1_metric.png")
        plt.legend()
        plt.close()

    else:
        fig = plt.plot(f1_list, label="f1 score")
        plt.ylim(0, 1)
        plt.savefig(args.output_path / "f1_metric.png")
        plt.close()

    if not args.on_cluster and args.experiment_str is None:
        experiment_str = "experiment"
    elif args.experiment_str is not None:
        experiment_str = args.experiment_str


if __name__ == "__main__":
    training()