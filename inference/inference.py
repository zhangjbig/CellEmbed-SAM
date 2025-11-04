import os
import pandas as pd
from tqdm.auto import tqdm
import torch
from pathlib import Path
import argparse
import fastremap
import time
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, help="mode")
parser.add_argument("--host", type=str, help="host")
parser.add_argument("--port", type=int, help="port")

parser.add_argument("-d_p", "--data_path", type=str, default=r"../cellembed/datasets")

parser.add_argument("-o_f", "--output_folder", type=str, default="Results")
parser.add_argument("-m", "--model_str", type=str, default="cellembed", help="Model backbone to use")
parser.add_argument("-m_p", "--model_path", type=str, default=r"../cellembed/C_TNBC_2018")
parser.add_argument("-m_f", "--model_folder", type=str, default="Cellembed SAM")
parser.add_argument("-d", "--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument("-db", "--debug", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-save_ims", "--save_ims", default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-source', '--source_dataset', default=None, type=str)
parser.add_argument('-o_h', '--optimize_hyperparameters', default=False, type=lambda x: (str(x).lower() == 'true'), help="Optimize postprocessing parameters")
parser.add_argument('-tta', '--tta', default=False, type=lambda x: (str(x).lower() == 'true'), help="Test time augmentations")
parser.add_argument('-target', '--target_segmentation', default=None, type=str, help="Cells or nuclei or both? Accepts: C,N, NC")
parser.add_argument('-params', '--params', default="default", type=str, help="Either 'default' or 'best_params'")
parser.add_argument('-window', '--window_size', default=128, type=int)

parser.add_argument('-export_to_torchscript', '--export_to_torchscript', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-export_to_bioimageio', '--export_to_bioimageio', default=False, type=lambda x: (str(x).lower() == 'true'))


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
            if masks is None:
                print("[Warning] masks is None, skipping renumber")
            else:
                masks[masks > 0], _ = fastremap.renumber(masks[masks > 0], in_place=True)
                gt_masks.append(masks.astype(np.int16))

            pred_masks.append(lab.astype(np.int16))


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

#@timer
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
    time_dict = {'preprocessing': 0, 'model': 0, 'postprocessing': 0, 'torchscript': 0, 'combined': []}

    pred_masks = []
    gt_masks = []

    model.eval()

    #####
    #warmup
    #####
    imgs, _ = Augmenter.to_tensor(val_images[0], normalize=False)
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
            imgs = imgs.to(device)
            imgs, _ = Augmenter.normalize(imgs)
            time_dict["preprocessing"] += time.time() - start
            torch.cuda.synchronize()
            start = time.time()

            if not tta:
                imgs, pad = _instanseg_padding(imgs, extra_pad=0, min_dim=32, ensure_square=False)
                with torch.amp.autocast("cuda"):
                    pred = model(imgs[None,])

                pred = _recover_padding(pred, pad).squeeze(0)
                imgs = _recover_padding(imgs, pad).squeeze(0)
                torch.cuda.synchronize()

                model_time = time.time() - start
                time_dict["model"] += model_time

                start = time.time()

                if params is not None:
                    with torch.amp.autocast("cuda"):
                        lab = postprocessing_fn(pred, **params, window_size=parser_args.window_size)
                else:
                    with torch.amp.autocast("cuda"):
                        lab = postprocessing_fn(pred, img=imgs, window_size=parser_args.window_size)

                torch.cuda.synchronize()

                postprocessing_time = time.time() - start

                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({"time": model_time + postprocessing_time, "model_time": model_time,
                                              "postprocessing_time": postprocessing_time, "dimension": imgs.shape,
                                              "num_instances": len(torch.unique(lab) - 1)})

            else:
                if params is not None:
                    lab = method.TTA_postprocessing(imgs[None,], model, transforms, **params,
                                                    window_size=parser_args.window_size, device=device)
                else:
                    lab = method.TTA_postprocessing(imgs[None,], model, transforms, window_size=parser_args.window_size,
                                                    device=device)

            imgs = imgs.cpu().numpy()

            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            if isinstance(lab, torch.Tensor):
                lab = lab.cpu().numpy()

            count += 1

            lab, _ = fastremap.renumber(lab, in_place=True)

            pred_masks.append(lab.astype(np.int16))


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

    from cellembed.utils.utils import show_images, save_image_with_label_overlay
    from cellembed.utils.model_loader import load_model
    from cellembed.utils.augmentations import Augmentations

    parser_args = parser.parse_args()

    model, model_dict = load_model(path=parser_args.model_path, folder=parser_args.model_folder)

    data_path = Path(parser_args.data_path)
    os.environ["INSTANSEG_DATASET_PATH"] = str(parser_args.data_path)
    device = parser_args.device
    n_sigma = model_dict['n_sigma']

    model_path = Path(parser_args.model_path) / Path(parser_args.model_folder)

    parser_args.loss_function = model_dict['loss_function']
    if parser_args.source_dataset is None:
        parser_args.source_dataset = model_dict['source_dataset']

    if parser_args.model_str.lower() == "cellembed":
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
        raise NotImplementedError("Loss function not recognized", parser_args.loss_function)

    model.eval()
    model.to(device)

    output_path = "./inference_image"
    output_path = Path(output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if parser_args.save_ims:
        if not os.path.exists(output_path / "images"):
            os.mkdir(output_path / "images")

    torch.cuda.empty_cache()
    count = 0

    Augmenter = Augmentations(dim_in=model_dict['dim_in'], shape=None)
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    folder_path = './inference_image'

    image_paths = sorted([
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith('.png')
    ])

    val_data = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            label = None

            tensor_img, _ = Augmenter.to_tensor(img_np, label, normalize=True)

            processed = Augmenter.duplicate_grayscale_channels(tensor_img, label)

            val_data.append(processed)

        except Exception as e:
            print(f"Skipped corrupted image: {img_path}, error: {e}")

    val_images = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]

    if parser_args.optimize_hyperparameters:
        from cellembed.utils.AI_utils import optimize_hyperparameters

        params = optimize_hyperparameters(model, postprocessing_fn=method.postprocessing, val_images=val_images,
                                          val_labels=val_labels, verbose=True)
        pd.DataFrame.from_dict(params, orient='index').to_csv(output_path / "best_params.csv",
                                                              header=False)
    else:
        if parser_args.params == "default":
            params = None
        else:
            df = pd.read_csv(output_path / "best_params.csv", header=None)
            params = {row[0]: row[1] for row in df.values}

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
