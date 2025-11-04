import numpy as np
from stardist import matching
import fastremap

from typing import Union
def _robust_f1_mean_calculator(nan_list: Union[list, np.ndarray]):
    nan_list = np.array(nan_list)
    if len(nan_list) == 0:
        return np.nan
    elif np.isnan(nan_list).all():
        return np.nan
    else:
        return np.nanmean(nan_list)


def _robust_average_precision(labels, predicted, threshold):
    # Fix sparse labels (<0 values)
    for i in range(len(labels)):
        if labels[i].min() < 0 and not (labels[i] < 0).all():
            mask = labels[i] < 0
            labels[i][mask] = 0
            predicted[i][mask] = 0

    # Case 1: single-channel (cells or nuclei)
    if labels[0].shape[0] != 2:
        labels_np = []
        preds_np = []
        for i in range(len(labels)):
            if labels[i].min() >= 0 and labels[i].max() > 0:
                labels_np.append(labels[i].detach().cpu().numpy().astype(np.int32))
                preds_np.append(predicted[i].detach().cpu().numpy().astype(np.int32))

        if len(labels_np) == 0:
            return np.nan, np.nan

        stats = matching.matching_dataset(labels_np, preds_np, thresh=threshold, show_progress=False)
        f1i = [stat.f1 for stat in stats]

        return _robust_f1_mean_calculator(f1i), f1i[0]

    # Case 2: two-channel (nuclei + cells)
    else:
        f1is = []
        f1_05is = []
        for chan in range(2):  # 0 = nuclei, 1 = cells
            labels_np = []
            preds_np = []
            for j in range(len(labels)):
                if labels[j][chan].min() >= 0 and labels[j][chan].max() > 0:
                    l_np = fastremap.renumber(labels[j][chan].detach().cpu().numpy())[0].astype(np.int32)
                    p_np = fastremap.renumber(predicted[j][chan].detach().cpu().numpy())[0].astype(np.int32)
                    labels_np.append(l_np)
                    preds_np.append(p_np)

            if len(labels_np) == 0:
                f1is.append(np.nan)
                f1_05is.append(np.nan)
                continue

            stats = matching.matching_dataset(labels_np, preds_np, thresh=threshold, show_progress=False)
            f1i = [stat.f1 for stat in stats]

            f1is.append(_robust_f1_mean_calculator(f1i))
            f1_05is.append(f1i[0])

        return f1is, f1_05is

    



import pandas as pd
from tqdm import tqdm
def compute_and_export_metrics(gt_masks, pred_masks, output_path, target, return_metrics = False, show_progress = False, verbose = True):
    taus = [ 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching.matching_dataset(gt_masks, pred_masks, thresh=t, show_progress=False, by_image = False) for t in tqdm(taus, disable=not show_progress)]
    df_list = []

    for stat in stats:
        df_list.append(pd.DataFrame([stat]))
    df = pd.concat(df_list, ignore_index=True)

    mean_f1 = df[["thresh", "f1"]].iloc[:].mean()["f1"]
    mean_panoptic_quality = df[["thresh", "panoptic_quality"]].iloc[:].mean()["panoptic_quality"]
    panoptic_quality_05 = df[["thresh", "panoptic_quality"]].iloc[0]["panoptic_quality"]
    f1_05 = df[["thresh", "f1"]].iloc[0]["f1"]


    df["mean_f1"] = mean_f1
    df["f1_05"] = f1_05
    df["mean_PQ"] = mean_panoptic_quality
    df["SQ"] = panoptic_quality_05 / f1_05

    if verbose:
        print("Target:",target)
        print("Mean f1 score: ", mean_f1)
        print("f1 score at 0.5: ", f1_05)
        print("SQ: ", panoptic_quality_05 / f1_05)

    if return_metrics:
        return mean_f1, f1_05, panoptic_quality_05 / f1_05

    if output_path is not None:

        df.to_csv(output_path / str(target + "_matching_metrics.csv"))
