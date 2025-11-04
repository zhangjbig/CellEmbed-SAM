import os
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from pathlib import Path
import argparse
import fastremap
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

matplotlib.use('Agg')

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, help="mode")
parser.add_argument("--host", type=str, help="host")
parser.add_argument("--port", type=int, help="port")

parser.add_argument("-d_p", "--data_path", type=str, default=r"../datasets")
parser.add_argument("-o_f", "--output_folder", type=str, default="Results")

parser.add_argument("-m_p", "--model_path", type=str, default=r"../All_train")
parser.add_argument("-m_f", "--model_folder", type=str, default="CEmbed SAM")
parser.add_argument("-data", "--dataset", type=str, default="All_train", help="Name of the dataset to load")
parser.add_argument("-save_ims", "--save_ims", default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("-d", "--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument("-db", "--debug", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-source', '--source_dataset', default=None, type=str)
parser.add_argument('-o_h', '--optimize_hyperparameters', default=False, type=lambda x: (str(x).lower() == 'true'),help="Optimize postprocessing parameters")
parser.add_argument('-tta', '--tta', default=False, type=lambda x: (str(x).lower() == 'true'),help="Test time augmentations")
parser.add_argument('-target', '--target_segmentation', default=None, type=str,help=" Cells or nuclei or both? Accepts: C,N, NC")
parser.add_argument('-params', '--params', default="default", type=str, help="Either 'default' or 'best_params'")
parser.add_argument('-window', '--window_size', default=128, type=int)
parser.add_argument('-set', '--test_set', default="Validation", type=str, help = "Validation or Test or Train")
parser.add_argument('-export_to_torchscript', '--export_to_torchscript', default=False,type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-export_to_bioimageio', '--export_to_bioimageio', default=False,type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-num_nuclei_classes', '--num_nuclei_classes', default=1, type=int, help="num_nuclei_classes")
parser.add_argument('-morpho', '--morphology_analysis', default=True, type=lambda x: (str(x).lower() == 'true'), help="Enable cell morphology metrics analysis")

def get_sliding_positions(total_size, patch_size, step):

    positions = list(range(0, total_size - patch_size + 1, step))
    if positions[-1] + patch_size < total_size:
        positions.append(total_size - patch_size)
    return positions

def sliding_window_center_inference(img, model, device, patch_size=256, center_crop=128, padding=128, batch_size=20, out_channels=None):

    C, H, W = img.shape

    if img.shape[1] > padding and img.shape[2] > padding:
        img_padded = F.pad(img, (padding, padding, padding, padding), mode='reflect')
    else:
        img_padded = F.pad(img, (padding, padding, padding, padding), mode='constant', value=0)

    H_pad, W_pad = img_padded.shape[1], img_padded.shape[2]

    pred_sum = torch.zeros((out_channels, H_pad, W_pad), dtype=torch.float32, device=device)
    count_map = torch.zeros((1, H_pad, W_pad), dtype=torch.float32, device=device)

    step = center_crop
    top_positions = get_sliding_positions(H_pad, patch_size, step)
    left_positions = get_sliding_positions(W_pad, patch_size, step)

    positions = [(top, left) for top in top_positions for left in left_positions]

    def crop_patch(pos):
        top, left = pos
        bottom = top + patch_size
        right = left + patch_size
        return img_padded[:, top:bottom, left:right], (top, left)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(crop_patch, positions))

    patches, positions = zip(*results)
    num_patches = len(patches)

    if num_patches == 0:
        return torch.zeros((out_channels, H, W), device=img.device)

    model.eval()
    with torch.no_grad():
        if num_patches < batch_size:
            batch_tensor = torch.stack(patches).to(device)
            with autocast():
                preds = model(batch_tensor)  # [N, C, patch_size, patch_size]


            for idx, (top, left) in enumerate(positions):
                pred = preds[idx]
                c_start = (patch_size - center_crop) // 2
                c_end = c_start + center_crop
                pred_crop = pred[:, c_start:c_end, c_start:c_end]

                top_dst = top + c_start
                left_dst = left + c_start
                bottom_dst = top_dst + center_crop
                right_dst = left_dst + center_crop

                pred_sum[:, top_dst:bottom_dst, left_dst:right_dst] += pred_crop
                count_map[:, top_dst:bottom_dst, left_dst:right_dst] += 1

        else:
            for i in range(0, num_patches, batch_size):
                batch_patches = patches[i:i + batch_size]
                batch_positions = positions[i:i + batch_size]
                batch_tensor = torch.stack(batch_patches).to(device)

                with autocast():
                    preds = model(batch_tensor)


                for idx, (top, left) in enumerate(batch_positions):
                    pred = preds[idx]
                    c_start = (patch_size - center_crop) // 2
                    c_end = c_start + center_crop
                    pred_crop = pred[:, c_start:c_end, c_start:c_end]

                    top_dst = top + c_start
                    left_dst = left + c_start
                    bottom_dst = top_dst + center_crop
                    right_dst = left_dst + center_crop

                    pred_sum[:, top_dst:bottom_dst, left_dst:right_dst] += pred_crop
                    count_map[:, top_dst:bottom_dst, left_dst:right_dst] += 1

    count_map[count_map == 0] = 1
    pred_avg = pred_sum / count_map
    pred_final = pred_avg[:, padding:padding + H, padding:padding + W].cpu()

    return pred_final


def instanseg_inference(val_images, val_labels, model, postprocessing_fn, device, parser_args, output_path, params=None,
                        instanseg=None, tta=False):

    if tta:
        import ttach as tta

        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 180, 90, 270]),
        ])

    from cellembed.utils.tiling import _instanseg_padding, _recover_padding

    count = 0
    time_dict = {'show_image': 0, 'preprocessing': 0, 'model': 0, 'postprocessing': 0, 'torchscript': 0, 'combined': []}

    pred_masks = []
    gt_masks = []

    model.eval()

    #####
    # warmup
    #####
    imgs, _ = Augmenter.to_tensor(val_images[0], normalize=False)
    H, W = imgs.shape[1], imgs.shape[2]

    if H > 256 or W > 256:
        imgs = imgs.to(device)
        imgs, _ = Augmenter.normalize(imgs)

        if not tta:

            pred = sliding_window_center_inference(imgs, model, device, patch_size=256, center_crop=128,
                                                   padding=128, batch_size=1, out_channels=method.dim_out).to(device)

            if params is not None:
                with autocast():
                    lab = postprocessing_fn(pred, **params, window_size=parser_args.window_size)
            else:
                with autocast():
                    lab = postprocessing_fn(pred, img=imgs, window_size=parser_args.window_size)

        else:

            if params is not None:
                with autocast():
                    lab = method.TTA_postprocessing(imgs[None,], model, transforms, **params,
                                                window_size=parser_args.window_size, device=device)
            else:
                with autocast():
                    lab = method.TTA_postprocessing(imgs[None,], model, transforms,
                                                window_size=parser_args.window_size,
                                                device=device)
    else:

        imgs = imgs.to(device)
        imgs, _ = Augmenter.normalize(imgs)

        with torch.no_grad():
            imgs, pad = _instanseg_padding(imgs, extra_pad=0, min_dim=32)

            with torch.amp.autocast("cuda"):
                pred = model(imgs[None,])

            pred = pred.float()
            pred = _recover_padding(pred, pad).squeeze(0)

            if params is not None:
                with torch.amp.autocast("cuda"):
                    lab = postprocessing_fn(pred, **params, window_size=parser_args.window_size)
            else:
                with torch.amp.autocast("cuda"):
                    lab = postprocessing_fn(pred, img=imgs, window_size=parser_args.window_size)

            lab = lab.cpu().numpy()

            if tta:
                lab = method.TTA_postprocessing(imgs[None,], model, transforms, device=device)


    with torch.no_grad():

        for imgs, masks in tqdm(zip(val_images, val_labels), total=len(val_images)):
            torch.cuda.synchronize()
            start = time.time()

            imgs, masks = Augmenter.to_tensor(imgs, masks, normalize=False)

            H, W = imgs.shape[1], imgs.shape[2]

            if H > 256 or W > 256:

                imgs = imgs.to(device)
                imgs, _ = Augmenter.normalize(imgs)

                time_dict["preprocessing"] += time.time() - start
                torch.cuda.synchronize()
                start = time.time()

                if not tta:

                    pred = sliding_window_center_inference(imgs, model, device, patch_size=256, center_crop=128,
                                                           padding=128, batch_size =1, out_channels=method.dim_out).to(device)

                    torch.cuda.synchronize()
                    model_time = time.time() - start
                    time_dict["model"] += model_time

                    start = time.time()

                    if params is not None:
                        with autocast():
                            lab = postprocessing_fn(pred, **params, window_size=parser_args.window_size)
                    else:
                        with autocast():
                            lab = postprocessing_fn(pred, img=imgs, window_size=parser_args.window_size)

                    torch.cuda.synchronize()
                    postprocessing_time = time.time() - start
                    time_dict["postprocessing"] += postprocessing_time

                    time_dict["combined"].append({"time": model_time + postprocessing_time, "model_time": model_time,
                                                  "postprocessing_time": postprocessing_time, "dimension": imgs.shape,
                                                  "num_instances": len(torch.unique(lab) - 1)})

                else:

                    if params is not None:
                        with autocast():
                            lab = method.TTA_postprocessing(imgs[None,], model, transforms, **params,
                                                        window_size=parser_args.window_size, device=device)
                    else:
                        with autocast():
                            lab = method.TTA_postprocessing(imgs[None,], model, transforms,
                                                        window_size=parser_args.window_size,
                                                        device=device)

            else:

                imgs = imgs.to(device)
                imgs, _ = Augmenter.normalize(imgs)

                time_dict["preprocessing"] += time.time() - start

                torch.cuda.synchronize()
                start = time.time()

                if not tta:

                    imgs, pad = _instanseg_padding(imgs, extra_pad=0, min_dim=32, ensure_square=False)

                    with autocast():
                        pred = model(imgs[None,])

                    pred = _recover_padding(pred, pad).squeeze(0)
                    imgs = _recover_padding(imgs, pad).squeeze(0)

                    torch.cuda.synchronize()

                    model_time = time.time() - start
                    time_dict["model"] += model_time

                    start = time.time()
                    model_single = None

                    if params is not None:
                        with autocast():
                            lab = postprocessing_fn(pred, **params, window_size=parser_args.window_size)
                    else:
                        with autocast():
                            lab = postprocessing_fn(pred, img=imgs, window_size=parser_args.window_size)

                    torch.cuda.synchronize()
                    postprocessing_time = time.time() - start

                    time_dict["postprocessing"] += postprocessing_time

                    time_dict["combined"].append({"time": model_time + postprocessing_time, "model_time": model_time,
                                                  "postprocessing_time": postprocessing_time, "dimension": imgs.shape,
                                                  "num_instances": len(torch.unique(lab) - 1)})

                else:
                    if params is not None:
                        with autocast():
                            lab = method.TTA_postprocessing(imgs[None,], model, transforms, **params,
                                                        window_size=parser_args.window_size, device=device)
                    else:
                        with autocast():
                            lab = method.TTA_postprocessing(imgs[None,], model, transforms, window_size=parser_args.window_size,
                                                        device=device)

            imgs = imgs.cpu().numpy()

            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            if isinstance(lab, torch.Tensor):
                lab = lab.cpu().numpy()

            count += 1

            lab, _ = fastremap.renumber(lab, in_place=True)
            masks[masks > 0], _ = fastremap.renumber(masks[masks > 0], in_place=True)

            pred_masks.append(lab.astype(np.int16))
            gt_masks.append(masks.astype(np.int16))

            if parser_args.save_ims:
                from cellembed.utils.augmentations import Augmentations
                augmenter = Augmentations()

                display = augmenter.colourize(torch.tensor(imgs), random_seed=1)[0]

                def overlay(img, gt, color=None):
                    return save_image_with_label_overlay(img, gt, return_image=True, alpha=0.8,
                                                         label_boundary_mode="thick", label_colors=color)

                show_images(overlay(display.numpy(), torch.tensor(lab)),
                            save_str=output_path / str("images/" "overlay" + str(count)))

    print("Time spent in preprocessing", time_dict["preprocessing"], "Time spent in model:", time_dict["model"],
          "Time spent in postprocessing:", time_dict["postprocessing"])

    return pred_masks, gt_masks, time_dict


if __name__ == "__main__":

    global method

    from cellembed.utils.utils import show_images, save_image_with_label_overlay, _move_channel_axis
    from cellembed.utils.model_loader import load_model
    from cellembed.utils.metrics import compute_and_export_metrics
    from cellembed.utils.augmentations import Augmentations

    parser_args = parser.parse_args()


    if parser_args.model_folder == "CEmbed SAM" or parser_args.model_folder == "Instanseg":

        model, model_dict = load_model(path=parser_args.model_path, folder=parser_args.model_folder)

        data_path = Path(parser_args.data_path)
        os.environ["INSTANSEG_DATASET_PATH"] = str(parser_args.data_path)
        device = parser_args.device
        n_sigma = model_dict['n_sigma']

        model_path = Path(parser_args.model_path) / Path(parser_args.model_folder)

        parser_args.model_str = model_dict['model_str']
        if parser_args.source_dataset is None:
            parser_args.source_dataset = model_dict['source_dataset']

        if parser_args.model_str.lower() == "instanseg_unet":
            from cellembed.utils.loss.instanseg_loss import InstanSeg
            method = InstanSeg(binary_loss_fn_str=model_dict["binary_loss_fn"], seed_loss_fn=model_dict["seed_loss_fn"],
                               n_sigma=model_dict["n_sigma"],
                               cells_and_nuclei=model_dict["cells_and_nuclei"], to_centre=model_dict["to_centre"],
                               window_size=parser_args.window_size, dim_coords=model_dict["dim_coords"],
                               feature_engineering_function=model_dict["feature_engineering"],
                               only_positive_labels=model_dict["only_positive_labels"])

            if parser_args.target_segmentation is None:
                parser_args.cells_and_nuclei = model_dict["cells_and_nuclei"]
                parser_args.target_segmentation = model_dict["target_segmentation"]

            else:
                if len(parser_args.target_segmentation) == 2:
                    parser_args.cells_and_nuclei = True
                else:
                    parser_args.cells_and_nuclei = False

            parser_args.pixel_size = model_dict["pixel_size"]

            import math

            if math.isnan(parser_args.pixel_size):
                parser_args.pixel_size = None

            method.initialize_pixel_classifier(model)


            def loss_fn(*args, **kwargs):
                return method.forward(*args, **kwargs)


            def get_labels(pred, **kwargs):
                return method.postprocessing(pred, **kwargs, device=device, max_seeds=10000)


            dim_out = method.dim_out

        elif parser_args.model_str.lower() == "cellembed":
            from cellembed.utils.loss.cellembed_loss import InstanSeg

            method = InstanSeg(binary_loss_fn_str=model_dict["binary_loss_fn"], seed_loss_fn=model_dict["seed_loss_fn"],
                               n_sigma=model_dict["n_sigma"],
                               cells_and_nuclei=model_dict["cells_and_nuclei"], to_centre=model_dict["to_centre"],
                               window_size=parser_args.window_size, dim_coords=model_dict["dim_coords"],
                               feature_engineering_function=model_dict["feature_engineering"])

            if parser_args.target_segmentation is None:
                parser_args.cells_and_nuclei = model_dict["cells_and_nuclei"]
                parser_args.target_segmentation = model_dict["target_segmentation"]

            else:
                if len(parser_args.target_segmentation) == 2:
                    parser_args.cells_and_nuclei = True
                else:
                    parser_args.cells_and_nuclei = False

            parser_args.pixel_size = model_dict["pixel_size"]

            import math

            if math.isnan(parser_args.pixel_size):
                parser_args.pixel_size = None

            method.initialize_pixel_classifier(model)

            def loss_fn(*args, **kwargs):
                return method.forward(*args, **kwargs)

            def get_labels(pred, **kwargs):
                return method.postprocessing_seed(pred, **kwargs, device=device, max_seeds=10000)


            dim_out = method.dim_out

        else:
            raise NotImplementedError("Model str not recognized", parser_args.model_str)

    elif parser_args.model_folder == "CellPose SAM":

        from othermodel.cellposesam.vit_sam import Transformer

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        pretrained_encoder = os.path.join(BASE_DIR, "sam_vit_l.pth")

        model = Transformer(backbone="vit_l", checkpoint=pretrained_encoder)
        pretrained_encoder = f'{parser_args.model_path}/{parser_args.model_folder}/model_weights.pth'
        state_dict = torch.load(str(pretrained_encoder), map_location="cpu")

        from cellembed.utils.model_loader import remove_module_prefix_from_dict
        state_dict['model_state_dict'] = remove_module_prefix_from_dict(state_dict['model_state_dict'])

        msg = model.load_state_dict(state_dict['model_state_dict'], strict=True)
        print(f"Loading checkpoint: {msg}")

    elif parser_args.model_folder == "Micro SAM":

        from segment_anything import sam_model_registry

        abbreviated_model_type = "vit_l"
        sam = sam_model_registry[abbreviated_model_type]()
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        pretrained_encoder = os.path.join(BASE_DIR, "sam_vit_l.pth")
        state_dict = torch.load(str(pretrained_encoder), map_location="cpu")
        msg = sam.image_encoder.load_state_dict(state_dict, strict=True)
        print(f"Loading checkpoint: {msg}")

        from othermodel.microsam import UNETR

        # Get the UNETR.
        out_channels = 3
        model = UNETR(
            img_size=256,
            backbone="sam",
            encoder=sam.image_encoder,
            out_channels=out_channels,
            use_sam_stats=True,
            final_activation="Sigmoid",
            use_skip_connection=False,
            use_conv_transpose=False,
            resize_input=False,
        )

        pretrained_encoder = f'{parser_args.model_path}/{parser_args.model_folder}/model_weights.pth'
        state_dict = torch.load(str(pretrained_encoder), map_location="cpu")

        from cellembed.utils.model_loader import remove_module_prefix_from_dict
        msg = model.load_state_dict(remove_module_prefix_from_dict(state_dict['model_state_dict']), strict=True)
        print(f"Loading checkpoint: {msg}")

    elif parser_args.model_folder == "CellViT":

        from othermodel.cellvit.cellvit import CellViTSAM
        model_class = CellViTSAM
        pretrained_encoder = f'{parser_args.model_path}/{parser_args.model_folder}/model_weights.pth'

        num_nuclei_classes = 1
        num_tissue_classes = 2
        vit_structure = "SAM-L"
        regression_loss = False

        model = model_class(
            model_path=pretrained_encoder,
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            vit_structure=vit_structure,
            regression_loss=regression_loss,
        )

        state_dict = torch.load(str(pretrained_encoder), map_location="cpu")
        from cellembed.utils.model_loader import remove_module_prefix_from_dict

        msg = model.load_state_dict(remove_module_prefix_from_dict(state_dict['model_state_dict']), strict=True)
        print(f"Loading checkpoint: {msg}")

    elif parser_args.model_folder == "Mediar":
        from othermodel.mediar import models
        model = models.MEDIARFormer(encoder_name='mit_b5',
                                    encoder_weights='imagenet',
                                    decoder_channels=[1024, 512, 256, 128, 64],
                                    decoder_pab_channels=256,
                                    in_channels=3,
                                    classes=3)

        pretrained_encoder = f'{parser_args.model_path}/{parser_args.model_folder}/model_weights.pth'
        state_dict = torch.load(str(pretrained_encoder), map_location="cpu")
        from cellembed.utils.model_loader import remove_module_prefix_from_dict
        state_dict['model_state_dict'] = remove_module_prefix_from_dict(state_dict['model_state_dict'])

        msg = model.load_state_dict(state_dict['model_state_dict'], strict=True)
        print(f"Loading checkpoint: {msg}")

    elif parser_args.model_folder == "CellPose3":
        from othermodel.cellpose.resnet_torch import CPnet
        nbase = [3, 32, 64, 128, 256]
        model = CPnet(nbase=nbase, nout=3, sz=3, mkldnn=False, max_pool=True, diam_mean=30)
        pretrained_encoder = f'{parser_args.model_path}/{parser_args.model_folder}/model_weights.pth'
        state_dict = torch.load(str(pretrained_encoder), map_location="cpu")
        from cellembed.utils.model_loader import remove_module_prefix_from_dict
        state_dict['model_state_dict'] = remove_module_prefix_from_dict(state_dict['model_state_dict'])

        msg = model.load_state_dict(state_dict['model_state_dict'], strict=True)
        print(f"Loading checkpoint: {msg}")

    if not parser_args.model_folder == "CEmbed SAM" or parser_args.model_folder == "Instanseg":
        from cellembed.utils.model_loader import read_model_args_from_csv_cell
        model_dict = read_model_args_from_csv_cell(path=parser_args.model_path, folder=parser_args.model_folder)

        data_path = Path(parser_args.data_path)
        os.environ["INSTANSEG_DATASET_PATH"] = str(parser_args.data_path)
        device = parser_args.device
        n_sigma = model_dict['n_sigma']

        model_path = Path(parser_args.model_path) / Path(parser_args.model_folder)

        if parser_args.source_dataset is None:
            parser_args.source_dataset = model_dict['source_dataset']

        parser_args.cells_and_nuclei = model_dict["cells_and_nuclei"]
        parser_args.target_segmentation = model_dict["target_segmentation"]

        if len(parser_args.target_segmentation) == 2:
            parser_args.cells_and_nuclei = True
        else:
            parser_args.cells_and_nuclei = False

        parser_args.pixel_size = model_dict["pixel_size"]

        import math

        if math.isnan(parser_args.pixel_size):
            parser_args.pixel_size = None


    model.eval()
    if "inference_folder" not in parser_args or parser_args.inference_folder is None:
        from cellembed.utils.data_loader import _read_images_from_pth
        val_images, val_labels, val_meta = _read_images_from_pth(args=parser_args, sets=[parser_args.test_set],
                                                                 dataset=parser_args.dataset)

    else:
        from cellembed.utils.data_loader import _read_images_from_path
        val_images, val_labels = _read_images_from_path(sets=[parser_args.test_set])

    datasets_str = np.unique([item['parent_dataset'] for item in val_meta])
    print("Datasets used:", datasets_str)

    model.to(device)

    output_path = model_path / parser_args.output_folder
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if parser_args.save_ims:
        if not os.path.exists(output_path / "images"):
            os.mkdir(output_path / "images")

    torch.cuda.empty_cache()
    count = 0

    Augmenter = Augmentations(dim_in=model_dict['dim_in'], shape=None)

    val_data = [Augmenter.duplicate_grayscale_channels(*Augmenter.to_tensor(img, label, normalize= True)) for img, label in
            zip(val_images, val_labels)]


    if parser_args.pixel_size is not None and parser_args.pixel_size != "None":
        print("Warning, rescaling image and ground truth labels to pixel size:", parser_args.pixel_size,
              "microns/pixel")

        for i, (img, label) in enumerate(val_data):
            if "pixel_size" not in val_meta[i].keys():
                val_meta[i]["pixel_size"] = 0.5

            elif val_meta[i]["pixel_size"] == "pixel_size":
                val_meta[i]["pixel_size"] = 0.5  #bug in mesmer dataset

        val_data = [Augmenter.torch_rescale(img, label, current_pixel_size=val_meta[i]['pixel_size'],
                                            requested_pixel_size=parser_args.pixel_size,
                                            modality=val_meta[i]["image_modality"], crop=False) for i, (img, label) in
                    enumerate(val_data)]


    val_images = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]

    from cellembed.utils.utils import count_instances

    freq = np.array([count_instances(label) for label in val_labels])
    area = np.array(
        [(len(label[label > 0].flatten())) / f for f, label in
         zip(freq, val_labels)])

    print("Found:", sum(freq), "instances, across", len(freq), "images.", "Median area:", np.median(area), "pixels")

    if parser_args.optimize_hyperparameters:
        from cellembed.utils.AI_utils import optimize_hyperparameters

        params = optimize_hyperparameters(model, postprocessing_fn=method.postprocessing_seed, val_images=val_images,
                                          val_labels=val_labels, verbose=True)
        pd.DataFrame.from_dict(params, orient='index').to_csv(output_path / "best_params.csv",
                                                              header=False)

    else:
        if parser_args.params == "default":
            params = None
        else:
            df = pd.read_csv(output_path / "best_params.csv", header=None)
            params = {row[0]: row[1] for row in df.values}

    # params["window_size"] = parser_args.window_size
    instanseg = None
    if parser_args.export_to_torchscript:
        from cellembed.utils.utils import export_to_torchscript

        print("Exporting model to torchscript")
        export_to_torchscript(parser_args.model_folder)
        instanseg = torch.jit.load("../torchscripts/" + parser_args.model_folder + ".pt")
    if parser_args.export_to_bioimageio:
        print("Exporting model to bioimageio")
        from cellembed.utils.create_bioimageio_model import export_bioimageio

        instanseg = torch.jit.load("../torchscripts/" + parser_args.model_folder + ".pt")
        export_bioimageio(instanseg, deepimagej=True, test_img_path="../examples/HE_example.tif",
                          model_name=parser_args.model_folder)


    if parser_args.model_folder == "CEmbed SAM" or parser_args.model_folder == "Instanseg":
        pred_masks, gt_masks, time_dict = instanseg_inference(val_images,
                                                              val_labels,
                                                              model,
                                                              postprocessing_fn=get_labels,
                                                              device=device,
                                                              parser_args=parser_args,
                                                              output_path=output_path,
                                                              params=params,
                                                              instanseg=instanseg,
                                                              tta=parser_args.tta)

    elif parser_args.model_folder == "CellPose SAM" or parser_args.model_folder == "CellPose3":
        from cellembed.scripts.inference import cellposesam_inference
        pred_masks, gt_masks, time_dict = cellposesam_inference(val_images,
                                                              val_labels,
                                                              model,
                                                              device=device,
                                                              save_ims=parser_args.save_ims,
                                                              output_path=output_path)


    elif parser_args.model_folder == "Micro SAM":

        from cellembed.scripts.inference import microsam_inference
        pred_masks, gt_masks, time_dict = microsam_inference(val_images,
                                                                val_labels,
                                                                model,
                                                                device=device,
                                                                save_ims=parser_args.save_ims,
                                                                output_path=output_path)

    elif parser_args.model_folder == "CellViT":

        from cellembed.scripts.inference import cellvit_inference
        pred_masks, gt_masks, time_dict = cellvit_inference(val_images,
                                                             val_labels,
                                                             model,
                                                             device=device,
                                                             save_ims=parser_args.save_ims,
                                                             output_path=output_path)


    elif parser_args.model_folder == "Mediar":

        from cellembed.scripts.inference import mediar_inference
        pred_masks, gt_masks, time_dict = mediar_inference(val_images,
                                                            val_labels,
                                                            model,
                                                            device=device,
                                                            save_ims=parser_args.save_ims,
                                                            output_path=output_path)


    pd.DataFrame(time_dict['combined']).to_csv(output_path / "timing_dict.csv", header=True)

    # morphology_analysis
    if parser_args.morphology_analysis:

        dataset_name = parser_args.dataset

        from stardist import matching
        from stardist.matching import precision, accuracy, recall, f1
        import seaborn as sns
        from matplotlib.colors import to_rgb
        from cellembed.utils.cellmorphology import CellImageAnalyzer

        f1_list = []
        precision_list = []
        recall_list = []
        accuracy_list = []
        error_list = []

        fn_list = []
        fp_list = []
        tp_list = []

        n_true_list = []
        n_pred_list = []

        mean_true_score_list = []
        mean_matched_score_list = []
        panoptic_quality_list = []

        gt_morpho_dfs = []

        for gt, pred in zip(gt_masks, pred_masks):
            stats = matching.matching_dataset([gt], [pred], thresh=0.5, show_progress=False, by_image=False)
            f1_list.append(stats.f1)
            accuracy_list.append(stats.accuracy)
            precision_list.append(stats.precision)
            recall_list.append(stats.recall)

            fn_list.append(stats.fn)
            fp_list.append(stats.fp)
            tp_list.append(stats.tp)

            denominator = stats.tp + stats.fn
            if denominator == 0:
                error = 0.0
            else:
                error = (stats.fn + stats.fp) / denominator

            error_list.append(error)

            n_true_list.append(stats.n_true)
            n_pred_list.append(stats.n_pred)

            mean_true_score_list.append(stats.mean_true_score)
            mean_matched_score_list.append(stats.mean_matched_score)
            panoptic_quality_list.append(stats.panoptic_quality)


            gt_image = np.squeeze(gt).astype(np.uint16)

            # morphology_analysis
            analyzer = CellImageAnalyzer(image_array=gt_image)
            analyzer.compute_contours_and_stats()
            analyzer.calculate_morphology_stats()
            df = analyzer.generate_dataframe()
            mean_values = df.dropna().mean()
            gt_morpho_dfs.append(mean_values)

        eps = 1e-10
        n_MAE_list = np.abs(np.array(n_pred_list) - np.array(n_true_list)) / (np.array(n_true_list) + eps)

        df = pd.DataFrame({
            "F1-score": f1_list,
            "Tp": tp_list,
            "Fn": fn_list,
            "Fp": fp_list,
            "Accuracy": accuracy_list,
            "Precision": precision_list,
            "Recall": recall_list,
            "Error_rate": error_list,
            "n_true": n_true_list,
            "n_pred": n_pred_list,
            "n_MAE": n_MAE_list,
            "mean_true_score": mean_true_score_list,
            "mean_matched_score": mean_matched_score_list,
            "panoptic_quality": panoptic_quality_list
        })

        morpho_df = pd.DataFrame(gt_morpho_dfs)
        result_df = pd.concat([df.reset_index(drop=True), morpho_df.reset_index(drop=True)], axis=1)

        result_df.to_excel(output_path / f"{dataset_name}_per_image_metrics.xlsx", index=False)

        F1_score = f1(sum(tp_list), sum(fp_list), sum(fn_list))
        Precision = precision(sum(tp_list), sum(fp_list), sum(fn_list))
        Recall = recall(sum(tp_list), sum(fp_list), sum(fn_list))
        Accuracy = accuracy(sum(tp_list), sum(fp_list), sum(fn_list))
        N_true = sum(n_true_list)
        N_pred = sum(n_pred_list)

        print(f"Mean F1-score: {F1_score:.4f}")
        print(f"Mean MAE: {abs(N_pred - N_true) / N_true:.4f}")

        mean_pq = np.mean(panoptic_quality_list)
        print(f"Mean Panoptic Quality (Macro): {mean_pq:.4f}")

        weights = [n_true + n_pred for n_true, n_pred in zip(n_true_list, n_pred_list)]
        weighted_pq = (
            np.average(panoptic_quality_list, weights=weights)
            if np.sum(weights) > 0 else 0.0
        )
        print(f"Weighted Panoptic Quality: {weighted_pq:.4f}")

        metrics = ["F1-score", "Precision", "Recall", "Accuracy"]
        base_colors = ["skyblue", "lightgreen", "salmon", "orchid"]


        def darken_color(color, amount=0.6):

            c = np.array(to_rgb(color))
            c = c * amount
            c = np.clip(c, 0, 1)
            return c


        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df[metrics], palette=base_colors, width=0.15)

        plt.title("Distribution of Metrics (per Image)")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(True)

        global_vals = [F1_score, Precision, Recall, Accuracy]
        for i, base_color in enumerate(base_colors):
            mean_val = global_vals[i]
            dark_color = darken_color(base_color, amount=0.6)
            plt.hlines(mean_val, i - 0.05, i + 0.05, colors=[dark_color], linewidth=4)

        plt.tight_layout()
        plt.savefig(output_path / f"{dataset_name}_metrics_violin_plot.png", dpi=300)

        for i, base_color in enumerate(base_colors):
            mean_val = df[metrics[i]].mean()
            dark_color = darken_color(base_color, amount=0.6)
            plt.hlines(mean_val, i - 0.05, i + 0.05, colors=[dark_color], linewidth=4)

        plt.tight_layout()
        plt.savefig(output_path / "metrics_violin_plot.png", dpi=300)

        show_progress = False
        thresholds = np.arange(0.5, 1.05, 0.05)
        stats = [matching.matching_dataset(gt_masks, pred_masks, thresh=t, show_progress=False, by_image=False) for t in
                 tqdm(thresholds, disable=not show_progress)]
        df_list = []
        for stat in stats:
            df_list.append(pd.DataFrame([stat]))
        df_f1 = pd.concat(df_list, ignore_index=True)
        df_f1['error'] = (df_f1['fn'] + df_f1['fp']) / (df_f1['tp'] + df_f1['fn'])
        df_f1.to_excel(output_path / f"{dataset_name}_per_image_f1.xlsx", index=False)


    if parser_args.cells_and_nuclei:
        pred_nuclei_masks = [pred_mask[0] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask[0].min() >= 0]
        gt_nuclei_masks = [gt_mask[0] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask[0].min() >= 0]

        pred_cell_masks = [pred_mask[1] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask[1].min() >= 0]
        gt_cell_masks = [gt_mask[1] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask[1].min() >= 0]

        compute_and_export_metrics(gt_nuclei_masks, pred_nuclei_masks, output_path, target="Nuclei")
        compute_and_export_metrics(gt_cell_masks, pred_cell_masks, output_path, target="Cells")
    else:
        pred_masks = [(pred_mask).squeeze()[None] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if
                      gt_mask.min() >= 0]
        gt_masks = [(gt_mask).squeeze()[None] for gt_mask, pred_mask in zip(gt_masks, pred_masks) if gt_mask.min() >= 0]
        compute_and_export_metrics(gt_masks, pred_masks, output_path,
                                   target="Cells" if parser_args.target_segmentation == "C" else "Nuclei")

