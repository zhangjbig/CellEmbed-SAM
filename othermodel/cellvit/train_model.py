import numpy as np
from othermodel.cellvit.post_proc_cellvit import get_bounding_box
from scipy.ndimage import center_of_mass
import torch.nn.functional as F
from othermodel.cellvit.cellvit import DataclassHVStorage

def unpack_predictions(predictions, model, device):
    """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

    Args:
        predictions (dict): Dictionary with the following keys:
            * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
            * nuclei_binary_map: Logit output for binary nuclei prediction branch. Shape: (batch_size, 2, H, W)
            * hv_map: Logit output for hv-prediction. Shape: (batch_size, 2, H, W)
            * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)

    Returns:
        DataclassHVStorage: Processed network output
    """

    predictions["nuclei_binary_map"] = F.softmax(
        predictions["nuclei_binary_map"], dim=1
    )  # shape: (batch_size, 2, H, W)


    if "regression_map" not in predictions.keys():
        predictions["regression_map"] = None

    return predictions






import torch
def unpack_masks(masks, device):
    """Unpack the given masks. Main focus lays on reshaping and postprocessing masks to generate one dict

    Args:
        masks (dict): Required keys are:
            * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
            * nuclei_binary_map: Binary nuclei segmentations. Shape: (batch_size, H, W)
            * hv_map: HV-Map. Shape: (batch_size, 2, H, W)
            * nuclei_type_map: Nuclei instance-prediction and segmentation (not binary, each instance has own integer).
                Shape: (batch_size, num_nuclei_classes, H, W)

        tissue_types (list): List of string names of ground-truth tissue types

    Returns:
        DataclassHVStorage: GT-Results with matching shapes and output types
    """
    # get ground truth values, perform one hot encoding for segmentation maps
    gt_nuclei_binary_map_onehot = (
        F.one_hot(masks["nuclei_binary_map"], num_classes=2)
    ).type(
        torch.float32
    )  # background, nuclei

    # assemble ground truth dictionary
    gt = {
        "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(
            device
        ),  # shape: (batch_size, H, W, 2)
        "hv_map": masks["hv_map"].to(device),  # shape: (batch_size, H, W, 2)
        "instance_map": masks["instance_map"].to(
            device
        ),  # shape: (batch_size, H, W) -> each instance has one integer
    }
    if "regression_map" in masks:
        gt["regression_map"] = masks["regression_map"].to(device)


    return gt


def gen_instance_hv_map_single(inst_map: np.ndarray) -> np.ndarray:
    """
    Generate HV map (horizontal, vertical distances) for a single instance map.

    Args:
        inst_map (np.ndarray): 2D array of shape (H, W) with instance labels.

    Returns:
        np.ndarray: HV map of shape (2, H, W)
    """
    orig_inst_map = inst_map.copy()
    x_map = np.zeros_like(inst_map, dtype=np.float32)
    y_map = np.zeros_like(inst_map, dtype=np.float32)

    inst_ids = np.unique(orig_inst_map)
    inst_ids = inst_ids[inst_ids != 0]  # Remove background

    for inst_id in inst_ids:
        inst_mask = (orig_inst_map == inst_id).astype(np.uint8)
        y0, y1, x0, x1 = get_bounding_box(inst_mask)

        # Expand bbox by 2px with boundary check
        y0 = max(y0 - 2, 0)
        y1 = min(y1 + 2, orig_inst_map.shape[0])
        x0 = max(x0 - 2, 0)
        x1 = min(x1 + 2, orig_inst_map.shape[1])

        inst_crop = inst_mask[y0:y1, x0:x1]
        if inst_crop.shape[0] < 2 or inst_crop.shape[1] < 2:
            continue

        cy, cx = center_of_mass(inst_crop)
        cy = int(round(cy))
        cx = int(round(cx))

        xx = np.arange(inst_crop.shape[1]) - cx
        yy = np.arange(inst_crop.shape[0]) - cy
        inst_x, inst_y = np.meshgrid(xx, yy)

        inst_x[inst_crop == 0] = 0
        inst_y[inst_crop == 0] = 0

        inst_x = inst_x.astype(np.float32)
        inst_y = inst_y.astype(np.float32)

        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.min(inst_x[inst_x < 0])
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.max(inst_x[inst_x > 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.min(inst_y[inst_y < 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.max(inst_y[inst_y > 0])

        x_map[y0:y1, x0:x1][inst_crop > 0] = inst_x[inst_crop > 0]
        y_map[y0:y1, x0:x1][inst_crop > 0] = inst_y[inst_crop > 0]

    return np.stack([x_map, y_map], axis=0)  # (2, H, W)

def gen_instance_hv_map_batch(inst_map_batch: np.ndarray) -> np.ndarray:
    """
    Generate HV maps for a batch of instance maps.

    Args:
        inst_map_batch (np.ndarray): 3D array of shape (B, H, W)

    Returns:
        np.ndarray: 4D array of HV maps, shape (B, 2, H, W)
    """
    batch_hv = [gen_instance_hv_map_single(inst_map_batch[b]) for b in range(inst_map_batch.shape[0])]
    return np.stack(batch_hv, axis=0)  # (B, 2, H, W)





import numpy as np
import cv2
from scipy.ndimage import measurements, binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

def batch_hovernet_postproc(nuclei_binary_map: np.ndarray, hv_map: np.ndarray,
                             object_size: int = 10, ksize: int = 21) -> np.ndarray:
    nuclei_binary_map = np.array(nuclei_binary_map >= 0.5, dtype=np.int32)

    blb = measurements.label(nuclei_binary_map)[0]
    if np.max(blb) > 1:
        blb = remove_small_objects(blb, min_size=object_size)
    blb[blb > 0] = 1

    h_dir = cv2.normalize(hv_map[..., 0], None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(hv_map[..., 1], None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = 1 - cv2.normalize(cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize),
                               None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize),
                               None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = (overall >= 0.4).astype(np.int32)
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    if np.max(marker) > 1:
        marker = remove_small_objects(marker, min_size=object_size)

    instance_map = watershed(dist, markers=marker, mask=blb)

    return instance_map

def batch_hovernet_postproc_batch(nuclei_binary_map: np.ndarray, hv_map: np.ndarray,
                                   object_size: int = 10, ksize: int = 21) -> np.ndarray:

    assert nuclei_binary_map.shape[0] == hv_map.shape[0], "Batch size mismatch"
    batch_size = nuclei_binary_map.shape[0]
    instance_maps = []

    for i in range(batch_size):
        binary_map_i = nuclei_binary_map[i]
        hv_map_i = np.transpose(hv_map[i], (1, 2, 0))  # (2, H, W) → (H, W, 2)
        instance_map_i = batch_hovernet_postproc(binary_map_i, hv_map_i,
                                                 object_size=object_size, ksize=ksize)
        instance_maps.append(instance_map_i)

    return np.stack(instance_maps, axis=0)  # shape: (B, H, W)
