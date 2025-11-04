import os
from tqdm.auto import tqdm
import fastremap
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from cellembed.utils.utils import show_images, save_image_with_label_overlay


def get_sliding_positions(total_size, patch_size, step):
    positions = list(range(0, total_size - patch_size + 1, step))
    if positions[-1] + patch_size < total_size:
        positions.append(total_size - patch_size)
    return positions


def cellposesam_sliding_window_center_inference(img, model, device, patch_size=256, center_crop=128, padding=128, batch_size=20):
    C, H, W = img.shape
    out_channels = 3

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
            preds = model(batch_tensor)[0]

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

                preds = model(batch_tensor)[0]

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



def microsam_sliding_window_center_inference(img, model, device, patch_size=256, center_crop=128, padding=128, batch_size=20):
    """
    Sliding window inference that only keeps the central region of each patch
    to avoid unstable predictions near patch boundaries.

    Args:
        img: Input image [C, H, W]
        model: Trained model
        device: Device (e.g., 'cuda:0')
        patch_size: Size of each sliding window patch
        center_crop: Size of the central region to keep from each patch
        padding: Padding size around the input image
        batch_size: Batch size for inference

    Returns:
        pred_final: Final prediction without padding [C, H, W]
    """

    C, H, W = img.shape
    out_channels = 3

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
            preds = model(batch_tensor)

            for idx, (top, left) in enumerate(positions):
                pred = preds[idx]
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)
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

                preds = model(batch_tensor)

                for idx, (top, left) in enumerate(batch_positions):
                    pred = preds[idx]
                    if pred.dim() == 2:
                        pred = pred.unsqueeze(0)
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



def cellvit_sliding_window_center_inference(img, model, device, patch_size=256, center_crop=128, padding=128, batch_size=20):
    C, H, W = img.shape
    out_channels = 4

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
            preds = model(batch_tensor)
            preds = torch.cat([
                preds["nuclei_binary_map"],
                preds["hv_map"]
            ], dim=1)

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

                preds = model(batch_tensor)
                preds = torch.cat([
                    preds["nuclei_binary_map"],
                    preds["hv_map"]
                ], dim=1)

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


from othermodel.cellposesam import dynamics
def cellposesam_inference(val_images, val_labels, model, device, save_ims, output_path):

    from cellembed.utils.augmentations import Augmentations
    from cellembed.utils.tiling import _instanseg_padding, _recover_padding

    Augmenter = Augmentations(dim_in=3, shape=None)

    count = 0
    time_dict = {'preprocessing': 0, 'model': 0, 'postprocessing': 0, 'torchscript': 0, 'combined': []}

    pred_masks = []
    gt_masks = []

    model.eval()

    niter = 200
    cellprob_threshold = 0.0
    flow_threshold = 0.8
    min_size0 = 15
    max_size_fraction = 0.4
    resize = None

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

                pred = cellposesam_sliding_window_center_inference(
                    imgs, model, device, patch_size=256, center_crop=128,
                    padding=128, batch_size=20
                ).to(device)

                torch.cuda.synchronize()
                model_time = time.time() - start
                time_dict["model"] += model_time
                start = time.time()

                pred = pred.unsqueeze(0)
                cellprob = pred[:, -1, ...]
                dP = pred[:, -3:-1, ...].permute(1, 0, 2, 3)

                dP = dP.detach().cpu().numpy()
                cellprob = cellprob.detach().cpu().numpy()

                predicted_label = dynamics.resize_and_compute_masks(
                    dP[:, 0], cellprob[0],
                    niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, resize=resize,
                    min_size=min_size0, max_size_fraction=max_size_fraction,
                    device=device
                )

                lab = torch.from_numpy(predicted_label.astype(np.int32)).long()

                torch.cuda.synchronize()
                postprocessing_time = time.time() - start
                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({
                    "time": model_time + postprocessing_time,
                    "model_time": model_time,
                    "postprocessing_time": postprocessing_time,
                    "dimension": imgs.shape,
                    "num_instances": len(torch.unique(lab) - 1)
                })

            else:
                imgs = imgs.to(device)
                imgs, _ = Augmenter.normalize(imgs)

                time_dict["preprocessing"] += time.time() - start
                torch.cuda.synchronize()
                start = time.time()

                imgs, pad = _instanseg_padding(imgs, extra_pad=0, min_dim=32, ensure_square=False)
                pred = model(imgs[None,])[0]

                pred = _recover_padding(pred, pad).squeeze(0)
                imgs = _recover_padding(imgs, pad).squeeze(0)

                torch.cuda.synchronize()
                model_time = time.time() - start
                time_dict["model"] += model_time
                start = time.time()

                pred = pred.unsqueeze(0)
                cellprob = pred[:, -1, ...]
                dP = pred[:, -3:-1, ...].permute(1, 0, 2, 3)

                dP = dP.detach().cpu().numpy()
                cellprob = cellprob.detach().cpu().numpy()

                predicted_label = dynamics.resize_and_compute_masks(
                    dP[:, 0], cellprob[0],
                    niter=niter, cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold, resize=resize,
                    min_size=min_size0, max_size_fraction=max_size_fraction,
                    device=device
                )

                lab = torch.from_numpy(predicted_label.astype(np.int32)).long()

                torch.cuda.synchronize()
                postprocessing_time = time.time() - start
                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({
                    "time": model_time + postprocessing_time,
                    "model_time": model_time,
                    "postprocessing_time": postprocessing_time,
                    "dimension": imgs.shape,
                    "num_instances": len(torch.unique(lab) - 1)
                })

            lab = lab.unsqueeze(0)
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

            if save_ims:
                from cellembed.utils.augmentations import Augmentations
                augmenter = Augmentations()

                display = augmenter.colourize(torch.tensor(imgs), random_seed=1)[0]

                def overlay(img, gt, color=None):
                    return save_image_with_label_overlay(
                        img, gt, return_image=True, alpha=0.8,
                        label_boundary_mode="thick", label_colors=color
                    )

                show_images(
                    overlay(display.numpy(), torch.tensor(lab)),
                    save_str=output_path / str("images/" "overlay" + str(count))
                )



    print(
        "Time spent in preprocessing", time_dict["preprocessing"],
        "Time spent in model:", time_dict["model"],
        "Time spent in postprocessing:", time_dict["postprocessing"]
    )

    return pred_masks, gt_masks, time_dict


def microsam_inference(val_images, val_labels, model, device, save_ims, output_path):
    from cellembed.utils.augmentations import Augmentations
    from cellembed.utils.tiling import _instanseg_padding, _recover_padding
    from othermodel.microsam import segmentation

    Augmenter = Augmentations(dim_in=3, shape=None)

    count = 0
    time_dict = {'preprocessing': 0, 'model': 0, 'postprocessing': 0, 'torchscript': 0, 'combined': []}

    pred_masks = []
    gt_masks = []

    model.eval()

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

                pred = microsam_sliding_window_center_inference(
                    imgs, model, device, patch_size=256, center_crop=128,
                    padding=128, batch_size=20
                ).to(device)

                torch.cuda.synchronize()
                model_time = time.time() - start
                time_dict["model"] += model_time

                start = time.time()

                pred = pred.unsqueeze(0)
                fg = pred[:, 0, ...]
                cdist = pred[:, 1, ...]
                bdist = pred[:, 2, ...]

                fg = fg.detach().cpu().numpy()
                cdist = cdist.detach().cpu().numpy()
                bdist = bdist.detach().cpu().numpy()

                min_dim = min(cdist.shape)
                safe_smoothing = min(2.0, (min_dim - 1) / 2.0)

                instances = segmentation.watershed_from_center_and_boundary_distances(
                    cdist, bdist, fg, min_size=50,
                    center_distance_threshold=0.5,
                    boundary_distance_threshold=0.6,
                    distance_smoothing=safe_smoothing
                )

                lab = torch.from_numpy(instances.astype(np.int32)).long()

                torch.cuda.synchronize()
                postprocessing_time = time.time() - start
                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({
                    "time": model_time + postprocessing_time,
                    "model_time": model_time,
                    "postprocessing_time": postprocessing_time,
                    "dimension": imgs.shape,
                    "num_instances": len(torch.unique(lab) - 1)
                })

            else:
                imgs = imgs.to(device)
                imgs, _ = Augmenter.normalize(imgs)

                time_dict["preprocessing"] += time.time() - start

                torch.cuda.synchronize()
                start = time.time()

                imgs, pad = _instanseg_padding(imgs, extra_pad=0, min_dim=32, ensure_square=False)
                pred = model(imgs[None,])

                pred = _recover_padding(pred, pad).squeeze(0)
                imgs = _recover_padding(imgs, pad).squeeze(0)

                torch.cuda.synchronize()

                model_time = time.time() - start
                time_dict["model"] += model_time

                start = time.time()

                pred = pred.unsqueeze(0)
                fg = pred[:, 0, ...]
                cdist = pred[:, 1, ...]
                bdist = pred[:, 2, ...]

                fg = fg.detach().cpu().numpy()
                cdist = cdist.detach().cpu().numpy()
                bdist = bdist.detach().cpu().numpy()

                min_dim = min(cdist.shape)
                safe_smoothing = min(2.0, (min_dim - 1) / 2.0)

                instances = segmentation.watershed_from_center_and_boundary_distances(
                    cdist, bdist, fg, min_size=50,
                    center_distance_threshold=0.5,
                    boundary_distance_threshold=0.6,
                    distance_smoothing=safe_smoothing
                )

                lab = torch.from_numpy(instances.astype(np.int32)).long()

                torch.cuda.synchronize()
                postprocessing_time = time.time() - start
                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({
                    "time": model_time + postprocessing_time,
                    "model_time": model_time,
                    "postprocessing_time": postprocessing_time,
                    "dimension": imgs.shape,
                    "num_instances": len(torch.unique(lab) - 1)
                })

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

            if save_ims:
                from cellembed.utils.augmentations import Augmentations
                augmenter = Augmentations()

                display = augmenter.colourize(torch.tensor(imgs), random_seed=1)[0]

                def overlay(img, gt, color=None):
                    return save_image_with_label_overlay(
                        img, gt, return_image=True, alpha=0.8,
                        label_boundary_mode="thick", label_colors=color
                    )

                show_images(
                    overlay(display.numpy(), torch.tensor(lab)),
                    save_str=output_path / str("images/" "overlay" + str(count))
                )


    print(
        "Time spent in preprocessing", time_dict["preprocessing"],
        "Time spent in model:", time_dict["model"],
        "Time spent in postprocessing:", time_dict["postprocessing"]
    )

    return pred_masks, gt_masks, time_dict


from othermodel.cellvit.train_model import unpack_predictions
from othermodel.cellvit.train_model import batch_hovernet_postproc_batch

def cellvit_inference(val_images, val_labels, model, device, save_ims, output_path):
    from cellembed.utils.augmentations import Augmentations
    from cellembed.utils.tiling import _instanseg_padding, _recover_padding

    Augmenter = Augmentations(dim_in=3, shape=None)

    count = 0
    time_dict = {'preprocessing': 0, 'model': 0, 'postprocessing': 0,
                 'torchscript': 0, 'combined': []}

    pred_masks = []
    gt_masks = []

    model.eval()

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

                pred = cellvit_sliding_window_center_inference(
                    imgs, model, device,
                    patch_size=256, center_crop=128,
                    padding=128, batch_size=20
                ).to(device)

                torch.cuda.synchronize()
                model_time = time.time() - start
                time_dict["model"] += model_time
                start = time.time()

                pred = pred.unsqueeze(0)
                nuclei_binary_map, hv_map = torch.split(pred, [2, 2], dim=1)
                predictions = {
                    "nuclei_binary_map": nuclei_binary_map,
                    "hv_map": hv_map
                }
                predictions = unpack_predictions(
                    predictions=predictions, model=model, device=device
                )
                lab = torch.tensor(
                    batch_hovernet_postproc_batch(
                        predictions['nuclei_binary_map'][:, 1, :, :].detach().cpu().numpy(),
                        predictions['hv_map'].detach().cpu().numpy()
                    )
                ).unsqueeze(1)

                torch.cuda.synchronize()
                postprocessing_time = time.time() - start
                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({
                    "time": model_time + postprocessing_time,
                    "model_time": model_time,
                    "postprocessing_time": postprocessing_time,
                    "dimension": imgs.shape,
                    "num_instances": len(torch.unique(lab) - 1)
                })

            else:
                imgs = imgs.to(device)
                imgs, _ = Augmenter.normalize(imgs)

                time_dict["preprocessing"] += time.time() - start
                torch.cuda.synchronize()
                start = time.time()

                imgs, pad = _instanseg_padding(
                    imgs, extra_pad=0, min_dim=32, ensure_square=False
                )
                pred = model(imgs[None,])

                pred = torch.cat([
                    pred["nuclei_binary_map"],
                    pred["hv_map"]
                ], dim=1)

                pred = _recover_padding(pred, pad).squeeze(0)
                imgs = _recover_padding(imgs, pad).squeeze(0)

                torch.cuda.synchronize()
                model_time = time.time() - start
                time_dict["model"] += model_time
                start = time.time()

                pred = pred.unsqueeze(0)
                nuclei_binary_map, hv_map = torch.split(pred, [2, 2], dim=1)
                predictions = {
                    "nuclei_binary_map": nuclei_binary_map,
                    "hv_map": hv_map
                }

                predictions = unpack_predictions(
                    predictions=predictions, model=model, device=device
                )
                lab = torch.tensor(
                    batch_hovernet_postproc_batch(
                        predictions['nuclei_binary_map'][:, 1, :, :].detach().cpu().numpy(),
                        predictions['hv_map'].detach().cpu().numpy()
                    )
                ).unsqueeze(1)

                torch.cuda.synchronize()
                postprocessing_time = time.time() - start
                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({
                    "time": model_time + postprocessing_time,
                    "model_time": model_time,
                    "postprocessing_time": postprocessing_time,
                    "dimension": imgs.shape,
                    "num_instances": len(torch.unique(lab) - 1)
                })

            lab = lab.squeeze(0)
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

            if save_ims:
                from cellembed.utils.augmentations import Augmentations
                augmenter = Augmentations()
                display = augmenter.colourize(torch.tensor(imgs), random_seed=1)[0]

                def overlay(img, gt, color=None):
                    return save_image_with_label_overlay(
                        img, gt, return_image=True, alpha=0.8,
                        label_boundary_mode="thick", label_colors=color
                    )

                show_images(
                    overlay(display.numpy(), torch.tensor(lab)),
                    save_str=output_path / str("images/" "overlay" + str(count))
                )


    print("Time spent in preprocessing", time_dict["preprocessing"],
          "Time spent in model:", time_dict["model"],
          "Time spent in postprocessing:", time_dict["postprocessing"])

    return pred_masks, gt_masks, time_dict




def mediar_inference(val_images, val_labels, model, device, save_ims, output_path):
    from cellembed.utils.augmentations import Augmentations
    from cellembed.utils.tiling import _instanseg_padding, _recover_padding

    from othermodel.mediar.core.MEDIAR.utils import labels_to_flows, compute_masks

    def test_sigmoid(z):
        """Sigmoid function for numpy arrays"""
        return 1 / (1 + np.exp(-z))


    Augmenter = Augmentations(dim_in=3, shape=None)

    count = 0
    time_dict = {'preprocessing': 0, 'model': 0, 'postprocessing': 0, 'torchscript': 0, 'combined': []}

    pred_masks = []
    gt_masks = []

    model.eval()

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

                pred = microsam_sliding_window_center_inference(
                    imgs, model, device, patch_size=256, center_crop=128,
                    padding=128, batch_size=20
                ).to(device)

                torch.cuda.synchronize()
                model_time = time.time() - start
                time_dict["model"] += model_time
                start = time.time()

                pred = pred.squeeze(0).detach().cpu().numpy()
                gradflows, cellprob = pred[:2], test_sigmoid(pred[-1])
                label, _ = compute_masks(gradflows, cellprob, use_gpu=True, device=device)
                lab = torch.from_numpy(label.astype(np.int32)).long().unsqueeze(0)

                torch.cuda.synchronize()
                postprocessing_time = time.time() - start
                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({
                    "time": model_time + postprocessing_time,
                    "model_time": model_time,
                    "postprocessing_time": postprocessing_time,
                    "dimension": imgs.shape,
                    "num_instances": len(torch.unique(lab) - 1)
                })

            else:
                imgs = imgs.to(device)
                imgs, _ = Augmenter.normalize(imgs)

                time_dict["preprocessing"] += time.time() - start
                torch.cuda.synchronize()
                start = time.time()

                imgs, pad = _instanseg_padding(imgs, extra_pad=0, min_dim=32, ensure_square=False)
                pred = model(imgs[None,])
                pred = _recover_padding(pred, pad).squeeze(0)
                imgs = _recover_padding(imgs, pad).squeeze(0)

                torch.cuda.synchronize()
                model_time = time.time() - start
                time_dict["model"] += model_time
                start = time.time()

                pred = pred.squeeze(0).detach().cpu().numpy()
                gradflows, cellprob = pred[:2], test_sigmoid(pred[-1])
                label, _ = compute_masks(gradflows, cellprob, use_gpu=True, device=device)
                lab = torch.from_numpy(label.astype(np.int32)).long().unsqueeze(0)

                torch.cuda.synchronize()
                postprocessing_time = time.time() - start
                time_dict["postprocessing"] += postprocessing_time

                time_dict["combined"].append({
                    "time": model_time + postprocessing_time,
                    "model_time": model_time,
                    "postprocessing_time": postprocessing_time,
                    "dimension": imgs.shape,
                    "num_instances": len(torch.unique(lab) - 1)
                })

            lab = lab

            from scipy.ndimage import binary_fill_holes
            lab_np = label.copy()
            ids = np.unique(lab_np)
            ids = ids[ids != 0]

            for i in ids:
                mask = (lab_np == i)
                filled = binary_fill_holes(mask)
                lab_np[mask] = 0
                lab_np[filled] = i

            label = lab_np
            lab = torch.from_numpy(label.astype(np.int32)).long().unsqueeze(0)

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

            if save_ims:
                from cellembed.utils.augmentations import Augmentations
                augmenter = Augmentations()
                display = augmenter.colourize(torch.tensor(imgs), random_seed=1)[0]

                def overlay(img, gt, color=None):
                    return save_image_with_label_overlay(
                        img, gt, return_image=True, alpha=0.8,
                        label_boundary_mode="thick", label_colors=color
                    )

                show_images(
                    overlay(display.numpy(), torch.tensor(lab)),
                    save_str=output_path / str("images/" "overlay" + str(count))
                )


    print(
        "Time spent in preprocessing", time_dict["preprocessing"],
        "Time spent in model:", time_dict["model"],
        "Time spent in postprocessing:", time_dict["postprocessing"]
    )

    return pred_masks, gt_masks, time_dict


