import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from cellembed.utils.metrics import _robust_average_precision, _robust_f1_mean_calculator

from cellembed.utils.augmentations import Augmentations
import time

from cellembed.utils.utils import show_images
import warnings


global_step = 0
def train_epoch(train_model, 
                train_device, 
                train_dataloader, 
                train_loss_fn, 
                train_optimizer, 
                args,
                ):
    
    global global_step
    start = time.time()
    train_model.train()
    train_loss = []
    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):

        image_batch = image_batch.to(train_device)
        labels = labels_batch.to(train_device)
        output = train_model(image_batch)
        loss, _, _ = train_loss_fn(output, labels.clone())
        loss = loss.mean()

        train_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)

        train_optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

    return np.mean(train_loss), end - start

def train_epoch_cellembed(train_model,
                train_device,
                train_dataloader,
                train_loss_fn,
                train_optimizer,
                args,
                ):
    global global_step
    start = time.time()
    train_model.train()
    train_loss = []

    torch.backends.cudnn.enabled = False

    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):
        image_batch = image_batch.to(train_device)

        labels = labels_batch.to(train_device)
        output = train_model(image_batch)

        loss, _, _  = train_loss_fn(output, labels.clone())
        loss = loss.mean()

        train_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)

        train_optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

    return np.mean(train_loss), end - start


def train_epoch_cellposesam(train_model,
                train_device,
                train_dataloader,
                train_optimizer,
                args,
                ):

    from torch import nn

    def _loss_fn_seg(lbl, y, device):
        """
        Calculates the loss function between true labels lbl and prediction y.

        Args:
            lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
            y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
            device (torch.device): Device on which the tensors are located.

        Returns:
            torch.Tensor: Loss value.

        """
        criterion = nn.MSELoss(reduction="mean")
        criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
        veci = 5. * lbl[:, -2:]
        loss = criterion(y[:, -3:-1], veci)
        loss /= 2.
        loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).float())
        loss = loss + loss2
        return loss


    global global_step
    start = time.time()
    train_model.train()
    train_loss = []

    torch.backends.cudnn.enabled = False

    from othermodel.cellposesam import dynamics
    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):

        image_batch = image_batch.to(train_device)

        train_files = None
        labels_list = [labels_batch[i].cpu().numpy() for i in range(labels_batch.shape[0])]
        train_labels = dynamics.labels_to_flows(labels_list, files=train_files, device=train_device)

        labels = torch.from_numpy(np.stack(train_labels)).float().to(train_device)

        output = train_model(image_batch)[0]

        loss = _loss_fn_seg(labels, output, train_device)

        loss = loss.mean()

        train_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)

        train_optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

    return np.mean(train_loss), end - start

def train_epoch_microsam(train_model,
                train_device,
                train_dataloader,
                train_optimizer,
                args,
                ):

    from othermodel.microsam.loss import DiceBasedDistanceLoss
    from othermodel.microsam.transform import PerObjectDistanceTransform
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
    )

    instance_seg_loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    global global_step
    start = time.time()
    train_model.train()
    train_loss = []

    torch.backends.cudnn.enabled = False

    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):

        image_batch = image_batch.to(train_device)
        labels = labels_batch.to(train_device)

        results = []
        for i in range(labels.shape[0]):
            single_img = labels[i, 0].cpu().numpy()
            dist = label_transform(single_img)
            results.append(dist)

        results = torch.from_numpy(np.stack(results, axis=0)).float().to(train_device)
        labels_for_unetr = results[:, 1:, ...]

        output = train_model(image_batch)

        loss = instance_seg_loss(output, labels_for_unetr)

        loss = loss.mean()

        train_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)

        train_optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

    return np.mean(train_loss), end - start

def train_epoch_cellvit(train_model,
                train_device,
                train_dataloader,
                train_optimizer,
                args,
                ):
    global global_step
    start = time.time()
    train_model.train()
    train_loss = []

    torch.backends.cudnn.enabled = False

    from othermodel.cellvit.train_model import unpack_predictions, unpack_masks, gen_instance_hv_map_batch
    from othermodel.cellvit.loss import compute_total_loss, retrieve_loss_fn

    loss_fn_dict = {
        "nuclei_binary_map": {
            "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 1},
            "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},
        },
        "hv_map": {
            "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 2.5},
            "msge": {"loss_fn": retrieve_loss_fn("msge_loss_maps"), "weight": 8},
        }
    }

    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):
        image_batch = image_batch.to(train_device)

        labels = labels_batch.to(train_device)
        inst_map = labels[:, 0, :, :].clone()
        np_map = labels[:, 0, :, :].clone()
        np_map[np_map > 0] = 1

        hv_map = torch.tensor(gen_instance_hv_map_batch(inst_map.cpu().numpy())).to(train_device)

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }

        output = train_model(image_batch)

        predictions = unpack_predictions(predictions=output, model=train_model, device=train_device)
        gt = unpack_masks(masks=masks, device=train_device)

        loss = compute_total_loss(predictions, gt, loss_fn_dict, train_device)

        loss = loss.mean()

        train_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)

        train_optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

    return np.mean(train_loss), end - start



def train_epoch_mediar(train_model,
                train_device,
                train_dataloader,
                train_optimizer,
                args,
                ):
    global global_step
    start = time.time()
    train_model.train()
    train_loss = []

    torch.backends.cudnn.enabled = False
    from othermodel.mediar.core.MEDIAR.utils import labels_to_flows
    import torch.nn as nn
    mse_loss = nn.MSELoss(reduction="mean")
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def mediar_criterion(outputs, labels_onehot_flows, device):
        """loss function between true labels and prediction outputs"""


        # Cell Recognition Loss
        cellprob_loss = bce_loss(
            outputs[:, -1],
            torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(device).float(),
        )

        # Cell Distinction Loss
        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(device)
        gradflow_loss = 0.5 * mse_loss(outputs[:, :2], 5.0 * gradient_flows)

        loss = cellprob_loss + gradflow_loss

        return loss

    for image_batch, labels_batch, _ in tqdm(train_dataloader, disable=args.on_cluster):

        image_batch = image_batch.to(train_device)
        labels = labels_batch.to(train_device)

        labels_onehot_flows = labels_to_flows(
            labels, use_gpu=True, device=train_device
        )

        outputs = train_model(image_batch)
        loss = mediar_criterion(outputs, labels_onehot_flows, device=train_device)

        loss = loss.mean()

        train_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.clip)

        train_optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    end = time.time()

    return np.mean(train_loss), end - start


def collate_fn(data):
    # data is of length batch size
    # data[0][0] is first image, data[0][1] os the first label

    # print(data[0][0].shape,len(data))
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    imgs, labels = zip(*data)
    lengths = [img.shape[0] for img in imgs]

    max_len = max(lengths)
    C, H, W = data[0][0].shape
    images = torch.zeros((len(data), max_len, H, W))
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)

    for i, img in enumerate(imgs):
        images[i, :len(img)] = img

    return images, labels, lengths.int()



global_step_test = 0
def test_epoch(test_model,
               test_device,
               test_dataloader,
               test_loss_fn,
               args,
               postprocessing_fn,
               method,
               iou_threshold,
               debug=False,
               save_str=None,
               save_bool=False,
               best_f1=None):
    global global_step_test
    start = time.time()

    test_model.eval()
    test_loss = []

    current_f1_list = []
    current_f1_05_list = []

    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)
            labels = labels_batch.to(test_device)
            output = test_model(image_batch)
            loss, _, _ = test_loss_fn(output, labels.clone())
            loss = loss.mean()

            test_loss.append(loss.detach().cpu().numpy())

            if type(output) == list:
                output = output[0]

            if labels.type() != 'torch.cuda.FloatTensor' and labels.type() != 'torch.FloatTensor':

                predicted_labels = torch.stack([postprocessing_fn(out) for out in output])

                f1i, f1_05 = _robust_average_precision(labels.clone(), predicted_labels.clone(),
                                                       threshold=iou_threshold)

                current_f1_list.append((f1i))
                current_f1_05_list.append((f1_05))

            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)
                current_f1_05_list.append(0)

            global_step_test += 1

    f1_array = np.array(current_f1_list)  # either N,2 or N,
    f1_05_array = np.array(current_f1_05_list)

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T
        f1_05_array = np.atleast_2d(f1_05_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)
    mean1_05_f1 = np.nanmean(f1_05_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    mean_05_f1 = _robust_f1_mean_calculator(mean1_05_f1)

    test_batches = list(tqdm(test_dataloader, disable=args.on_cluster))

    indices = [0, len(test_batches) // 2, len(test_batches) - 1]

    for i in indices:

        image_batch, labels_batch, _ = test_batches[i]

        image_batch = image_batch[0].unsqueeze(0).to(test_device)
        labels = labels_batch[0].unsqueeze(0).to(test_device)
        output = test_model(image_batch)

        current_save_str = f"{save_str}_tag{i}"

        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]

        labels_dst = labels[0]
        lab = postprocessing_fn(output[0])

        output[0][args.dim_coords + args.n_sigma - 1] = torch.sigmoid(output[0][args.dim_coords + args.n_sigma - 1])

        if len(args.target_segmentation) == 2:
            output[0][(args.dim_coords + args.n_sigma - 1) * 2 + 2] = torch.sigmoid(
                output[0][(args.dim_coords + args.n_sigma - 1) * 2 + 2])

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)

    end = time.time()
    return np.mean(test_loss) , mean1_f1, end - start, mean1_05_f1


def test_epoch_cellembed(test_model,
               test_device,
               test_dataloader,
               test_loss_fn,
               args,
               postprocessing_fn,
               method,
               iou_threshold,
               debug=False,
               save_str=None,
               save_bool=False,
               best_f1=None):
    global global_step_test
    start = time.time()

    test_model.eval()

    test_loss = []
    seed_Loss = []
    label_Loss = []
    current_f1_list = []
    current_f1_05_list = []


    with torch.no_grad():

        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)

            labels = labels_batch.to(test_device)
            output = test_model(image_batch)
            loss, seed_loss, label_loss = test_loss_fn(output, labels.clone())

            loss = loss.mean()
            seed_loss = seed_loss.mean()
            label_loss = label_loss.mean()

            test_loss.append(loss.detach().cpu().numpy())

            seed_Loss.append(seed_loss.detach().cpu().numpy())
            label_Loss.append(label_loss.detach().cpu().numpy())

            if labels.type() != 'torch.cuda.FloatTensor' and labels.type() != 'torch.FloatTensor':

                predicted_labels = torch.stack([postprocessing_fn(out) for out in output])

                f1i, f1_05 = _robust_average_precision(labels.clone(), predicted_labels.clone(),
                                               threshold=iou_threshold)

                current_f1_list.append((f1i))
                current_f1_05_list.append((f1_05))

            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)
                current_f1_05_list.append(0)

            global_step_test += 1

    f1_array = np.array(current_f1_list)  # either N,2 or N,
    f1_05_array = np.array(current_f1_05_list)

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T
        f1_05_array = np.atleast_2d(f1_05_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)
    mean1_05_f1 = np.nanmean(f1_05_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    mean_05_f1 = _robust_f1_mean_calculator(mean1_05_f1)

    test_batches = list(tqdm(test_dataloader, disable=args.on_cluster))
    indices = [0, len(test_batches) // 2, len(test_batches) - 1]

    for i in indices:

        image_batch, labels_batch, _ = test_batches[i]

        image_batch = image_batch[0].unsqueeze(0).to(test_device)
        labels = labels_batch[0].unsqueeze(0).to(test_device)
        output = test_model(image_batch)

        current_save_str = f"{save_str}_tag{i}"

        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]

        labels_dst = labels[0]
        lab = postprocessing_fn(output[0])


        output[0][args.dim_coords+args.n_sigma-1] = torch.sigmoid(output[0][args.dim_coords+args.n_sigma-1])

        if len(args.target_segmentation) == 2:
            output[0][(args.dim_coords+args.n_sigma-1)*2+2] = torch.sigmoid(output[0][(args.dim_coords+args.n_sigma-1)*2+2])

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)


    end = time.time()

    return np.mean(test_loss), np.mean(seed_Loss), np.mean(label_Loss), mean1_f1, end - start, mean1_05_f1

def test_epoch_cellposesam(test_model,
                       test_device,
                       test_dataloader,
                       args,
                       iou_threshold,
                       debug=False,
                       save_str=None,
                       save_bool=False,
                       best_f1=None):
    from torch import nn

    def _loss_fn_seg(lbl, y, device):
        """
        Calculates the loss function between true labels lbl and prediction y.

        Args:
            lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
            y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
            device (torch.device): Device on which the tensors are located.

        Returns:
            torch.Tensor: Loss value.
        """
        criterion = nn.MSELoss(reduction="mean")
        criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
        veci = 5. * lbl[:, -2:]
        loss = criterion(y[:, -3:-1], veci)
        loss /= 2.
        loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).float())
        loss = loss + loss2
        return loss

    global global_step_test
    start = time.time()

    # Set model to evaluation mode
    test_model.eval()

    # Initialize lists for test loss and F1 scores
    test_loss = []
    current_f1_list = []
    current_f1_05_list = []

    from othermodel.cellposesam import dynamics

    with torch.no_grad():
        # Loop over test dataloader
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)
            labels0 = labels_batch.to(test_device)

            train_files = None
            labels_list = [labels_batch[i].cpu().numpy() for i in range(labels_batch.shape[0])]
            train_labels = dynamics.labels_to_flows(labels_list, files=train_files, device=test_device)

            labels = torch.from_numpy(np.stack(train_labels)).float().to(test_device)

            output = test_model(image_batch)[0]
            loss = _loss_fn_seg(labels, output, test_device)
            loss = loss.mean()
            test_loss.append(loss.detach().cpu().numpy())

            # If labels are not float tensors, run postprocessing and compute F1
            if labels0.type() != 'torch.cuda.FloatTensor' and labels0.type() != 'torch.FloatTensor':
                predicted_labels = []
                cellprob = output[:, -1, ...]
                dP = output[:, -3:-1, ...].permute(1, 0, 2, 3)

                dP = dP.cpu().numpy()
                cellprob = cellprob.cpu().numpy()

                # Postprocess (flows to masks)
                niter = 200
                cellprob_threshold = 0.0
                flow_threshold = 0.8
                min_size0 = 15
                max_size_fraction = 0.4
                resize = None
                for i in range(cellprob.shape[0]):
                    predicted_label = dynamics.resize_and_compute_masks(
                        dP[:, i], cellprob[i],
                        niter=niter, cellprob_threshold=cellprob_threshold,
                        flow_threshold=flow_threshold, resize=resize,
                        min_size=min_size0, max_size_fraction=max_size_fraction,
                        device=test_device)
                    predicted_label = np.expand_dims(predicted_label, axis=0)
                    predicted_labels.append(predicted_label)

                predicted_labels = np.stack(predicted_labels, axis=0)
                predicted_labels = torch.from_numpy(predicted_labels.astype(np.int32)).long()

                f1i, f1_05 = _robust_average_precision(labels0.clone(), predicted_labels.clone(),
                                                       threshold=iou_threshold)
                current_f1_list.append((f1i))
                current_f1_05_list.append((f1_05))
            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)
                current_f1_05_list.append(0)

            global_step_test += 1

    # Convert F1 lists to arrays
    f1_array = np.array(current_f1_list)
    f1_05_array = np.array(current_f1_05_list)

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T
        f1_05_array = np.atleast_2d(f1_05_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)
    mean1_05_f1 = np.nanmean(f1_05_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    mean_05_f1 = _robust_f1_mean_calculator(mean1_05_f1)

    # Iterate dataloader once
    test_batches = list(tqdm(test_dataloader, disable=args.on_cluster))
    indices = [0, len(test_batches) // 2, len(test_batches) - 1]

    for i in indices:
        image_batch, labels_batch, _ = test_batches[i]
        image_batch = image_batch[0].unsqueeze(0).to(test_device)
        labels = labels_batch[0].unsqueeze(0).to(test_device)

        output = test_model(image_batch)[0]
        current_save_str = f"{save_str}_tag{i}"

        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]

        labels_dst = labels[0]
        cellprob = output[:, -1, ...]
        dP = output[:, -3:-1, ...].permute(1, 0, 2, 3)

        dP = dP.detach().cpu().numpy()
        cellprob = cellprob.detach().cpu().numpy()

        niter = 200
        cellprob_threshold = 0.0
        flow_threshold = 0.8
        min_size0 = 15
        max_size_fraction = 0.4
        resize = None

        predicted_label = dynamics.resize_and_compute_masks(
            dP[:, 0], cellprob[0],
            niter=niter, cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold, resize=resize,
            min_size=min_size0, max_size_fraction=max_size_fraction,
            device=test_device)

        lab = torch.from_numpy(predicted_label.astype(np.int32)).long()

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)

    end = time.time()

    return np.mean(test_loss), mean1_f1, end - start, mean1_05_f1


def test_epoch_microsam(test_model,
                       test_device,
                       test_dataloader,
                       args,
                       iou_threshold,
                       debug=False,
                       save_str=None,
                       save_bool=False,
                       best_f1=None):

    from othermodel.microsam.loss import DiceBasedDistanceLoss
    from othermodel.microsam.transform import PerObjectDistanceTransform
    from othermodel.microsam import segmentation

    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True,
    )

    instance_seg_loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    global global_step_test
    start = time.time()

    test_model.eval()

    test_loss = []
    current_f1_list = []
    current_f1_05_list = []

    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)
            labels = labels_batch.to(test_device)

            results = []
            for i in range(labels.shape[0]):
                single_img = labels[i, 0].cpu().numpy()
                dist = label_transform(single_img)
                results.append(dist)

            results = torch.from_numpy(np.stack(results, axis=0)).float().to(test_device)
            labels_for_unetr = results[:, 1:, ...]

            output = test_model(image_batch)

            loss = instance_seg_loss(output, labels_for_unetr)
            loss = loss.mean()
            test_loss.append(loss.detach().cpu().numpy())

            if labels.type() != 'torch.cuda.FloatTensor' and labels.type() != 'torch.FloatTensor':
                predicted_labels = []

                fg = output[:, 0, ...]
                cdist = output[:, 1, ...]
                bdist = output[:, 2, ...]

                fg = fg.cpu().numpy()
                cdist = cdist.cpu().numpy()
                bdist = bdist.cpu().numpy()

                for i in range(fg.shape[0]):
                    min_dim = min(cdist[i].shape)
                    safe_smoothing = min(2.0, (min_dim - 1) / 2.0)

                    instances = segmentation.watershed_from_center_and_boundary_distances(
                        center_distances=cdist[i],
                        boundary_distances=bdist[i],
                        foreground_map=fg[i],
                        min_size=50,
                        center_distance_threshold=0.5,
                        boundary_distance_threshold=0.6,
                        distance_smoothing=0.5
                    )
                    predicted_labels.append(np.expand_dims(instances, axis=0))

                predicted_labels = np.stack(predicted_labels, axis=0)
                predicted_labels = torch.from_numpy(predicted_labels.astype(np.int32)).long()

                f1i, f1_05 = _robust_average_precision(labels.clone(), predicted_labels.clone(),
                                                       threshold=iou_threshold)

                current_f1_list.append((f1i))
                current_f1_05_list.append((f1_05))
            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)
                current_f1_05_list.append(0)

            global_step_test += 1

    f1_array = np.array(current_f1_list)
    f1_05_array = np.array(current_f1_05_list)

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T
        f1_05_array = np.atleast_2d(f1_05_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)
    mean1_05_f1 = np.nanmean(f1_05_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    mean_05_f1 = _robust_f1_mean_calculator(mean1_05_f1)

    test_batches = list(tqdm(test_dataloader, disable=args.on_cluster))
    indices = [0, len(test_batches) // 2, len(test_batches) - 1]

    for i in indices:
        image_batch, labels_batch, _ = test_batches[i]

        image_batch = image_batch[0].unsqueeze(0).to(test_device)
        labels = labels_batch[0].unsqueeze(0).to(test_device)

        output = test_model(image_batch)[0].unsqueeze(0)

        current_save_str = f"{save_str}_tag{i}"

        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]

        labels_dst = labels[0]

        fg = output[:, 0, ...]
        cdist = output[:, 1, ...]
        bdist = output[:, 2, ...]

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

        lab = torch.from_numpy(instances.astype(np.int32)).long().unsqueeze(0)

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)

    end = time.time()
    return np.mean(test_loss), mean1_f1, end - start, mean1_05_f1





def test_epoch_cellvit(test_model,
               test_device,
               test_dataloader,
               args,
               iou_threshold,
               debug=False,
               save_str=None,
               save_bool=False,
               best_f1=None):

    from othermodel.cellvit.train_model import batch_hovernet_postproc_batch
    from othermodel.cellvit.train_model import unpack_predictions, unpack_masks, gen_instance_hv_map_batch
    from othermodel.cellvit.loss import compute_total_loss, retrieve_loss_fn

    global global_step_test
    start = time.time()

    test_model.eval()

    test_loss = []
    current_f1_list = []
    current_f1_05_list = []

    loss_fn_dict = {
        "nuclei_binary_map": {
            "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 1},
            "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},
        },
        "hv_map": {
            "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 2.5},
            "msge": {"loss_fn": retrieve_loss_fn("msge_loss_maps"), "weight": 8},
        }
    }

    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)

            labels = labels_batch.to(test_device)
            inst_map = labels[:, 0, :, :].clone()
            np_map = labels[:, 0, :, :].clone()
            np_map[np_map > 0] = 1

            hv_map = torch.tensor(gen_instance_hv_map_batch(inst_map.cpu().numpy())).to(test_device)

            masks = {
                "instance_map": torch.Tensor(inst_map).type(torch.int64),
                "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
                "hv_map": torch.Tensor(hv_map).type(torch.float32),
            }

            output = test_model(image_batch)

            predictions = unpack_predictions(predictions=output, model=test_model, device=test_device)
            gt = unpack_masks(masks=masks, device=test_device)

            loss = compute_total_loss(predictions, gt, loss_fn_dict, test_device)
            loss = loss.mean()
            test_loss.append(loss.detach().cpu().numpy())

            if labels.type() != 'torch.cuda.FloatTensor' and labels.type() != 'torch.FloatTensor':
                predicted_labels = torch.tensor(
                    batch_hovernet_postproc_batch(predictions['nuclei_binary_map'][:, 1, :, :].cpu().numpy(),
                                                  predictions['hv_map'].cpu().numpy())).unsqueeze(1)

                f1i, f1_05 = _robust_average_precision(labels.clone(), predicted_labels.clone(),
                                               threshold=iou_threshold)

                current_f1_list.append((f1i))
                current_f1_05_list.append((f1_05))
            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)
                current_f1_05_list.append(0)

            global_step_test += 1

    f1_array = np.array(current_f1_list)
    f1_05_array = np.array(current_f1_05_list)

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T
        f1_05_array = np.atleast_2d(f1_05_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)
    mean1_05_f1 = np.nanmean(f1_05_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    mean_05_f1 = _robust_f1_mean_calculator(mean1_05_f1)

    test_batches = list(tqdm(test_dataloader, disable=args.on_cluster))
    indices = [0, len(test_batches) // 2, len(test_batches) - 1]

    for i in indices:
        image_batch, labels_batch, _ = test_batches[i]

        image_batch = image_batch[0].unsqueeze(0).to(test_device)
        labels = labels_batch[0].unsqueeze(0).to(test_device)
        output = test_model(image_batch)

        predictions = unpack_predictions(predictions=output, model=test_model, device=test_device)
        predicted_labels = torch.tensor(
            batch_hovernet_postproc_batch(predictions['nuclei_binary_map'][:, 1, :, :].detach().cpu().numpy(),
                                          predictions['hv_map'].detach().cpu().numpy())).unsqueeze(1)

        output = torch.cat([
            predictions["nuclei_binary_map"],
            predictions["hv_map"]
        ], dim=1)

        current_save_str = f"{save_str}_tag{i}"

        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]

        labels_dst = labels[0]
        lab = predicted_labels[0]

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)

    end = time.time()
    return np.mean(test_loss), mean1_f1, end - start, mean1_05_f1


def test_epoch_mediar(test_model,
                       test_device,
                       test_dataloader,
                       args,
                       iou_threshold,
                       debug=False,
                       save_str=None,
                       save_bool=False,
                       best_f1=None):

    from othermodel.mediar.core.MEDIAR.utils import labels_to_flows, compute_masks
    import torch.nn as nn
    mse_loss = nn.MSELoss(reduction="mean")
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def mediar_criterion(outputs, labels_onehot_flows, device):
        """Loss function between true labels and prediction outputs"""

        cellprob_loss = bce_loss(
            outputs[:, -1],
            torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(device).float(),
        )

        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(device)
        gradflow_loss = 0.5 * mse_loss(outputs[:, :2], 5.0 * gradient_flows)

        loss = cellprob_loss + gradflow_loss
        return loss

    import numpy as np
    import torch

    def test_sigmoid(z):
        """Sigmoid function for numpy arrays"""
        return 1 / (1 + np.exp(-z))

    def process_single_output(output_b, device):
        output_b = output_b.detach().cpu().numpy()
        gradflows, cellprob = output_b[:2], test_sigmoid(output_b[-1])
        mask, _ = compute_masks(gradflows, cellprob, use_gpu=True, device=device)
        mask_tensor = torch.from_numpy(mask.astype(np.uint8)).unsqueeze(0)  # (1, H, W)
        return mask_tensor

    from concurrent.futures import ThreadPoolExecutor
    def test_post_process(outputs, labels=None, device='cuda'):
        """
        Efficient post-processing of model outputs into instance masks.

        Args:
            outputs: torch.Tensor of shape (B, C, H, W)
            labels: torch.Tensor of shape (B, 1, H, W) or None
            device: str

        Returns:
            pred_masks: torch.Tensor of shape (B, 1, H, W)
            labels_tensor: torch.Tensor of shape (B, 1, H, W) or None
        """
        if outputs.dim() != 4:
            raise ValueError(f"Expected 4D tensor for outputs, got {outputs.shape}")

        B = outputs.shape[0]
        pred_masks = [None] * B
        labels_list = [None] * B if labels is not None else None

        with ThreadPoolExecutor(max_workers=min(B, 40)) as executor:
            futures = [executor.submit(process_single_output, outputs[b], device) for b in range(B)]
            for b, f in enumerate(futures):
                pred_masks[b] = f.result()

        pred_masks = torch.stack(pred_masks, dim=0)  # (B, 1, H, W)

        if labels is not None:
            for b in range(B):
                label_b = labels[b]
                if label_b.dim() == 2:
                    label_b = label_b.unsqueeze(0)
                labels_list[b] = label_b.cpu()
            labels_tensor = torch.stack(labels_list, dim=0)  # (B, 1, H, W)
            return pred_masks, labels_tensor
        else:
            return pred_masks, None

    global global_step_test
    start = time.time()

    test_model.eval()

    test_loss = []
    current_f1_list = []
    current_f1_05_list = []

    with torch.no_grad():
        for image_batch, labels_batch, _ in tqdm(test_dataloader, disable=args.on_cluster):
            image_batch = image_batch.to(test_device)
            labels = labels_batch.to(test_device)

            labels_onehot_flows = labels_to_flows(
                labels, use_gpu=True, device=test_device
            )

            outputs = test_model(image_batch)
            loss = mediar_criterion(outputs, labels_onehot_flows, device=test_device)
            loss = loss.mean()
            test_loss.append(loss.detach().cpu().numpy())

            if labels.type() != 'torch.cuda.FloatTensor' and labels.type() != 'torch.FloatTensor':
                pred_masks, gt_labels = test_post_process(outputs, labels, device=test_device)

                predicted_labels = np.stack(pred_masks, axis=0)  # (B, 1, H, W)
                predicted_labels = torch.from_numpy(predicted_labels.astype(np.int32)).long()

                f1i, f1_05 = _robust_average_precision(labels.clone(), predicted_labels.clone(),
                                                       threshold=iou_threshold)

                current_f1_list.append((f1i))
                current_f1_05_list.append((f1_05))
            else:
                warnings.warn("Labels are of type float, not int. Not calculating F1.")
                current_f1_list.append(0)
                current_f1_05_list.append(0)

            global_step_test += 1

    f1_array = np.array(current_f1_list)
    f1_05_array = np.array(current_f1_05_list)

    if f1_array.ndim == 1:
        f1_array = np.atleast_2d(f1_array).T
        f1_05_array = np.atleast_2d(f1_05_array).T

    mean1_f1 = np.nanmean(f1_array, axis=0)
    mean1_05_f1 = np.nanmean(f1_05_array, axis=0)

    mean_f1 = _robust_f1_mean_calculator(mean1_f1)
    mean_05_f1 = _robust_f1_mean_calculator(mean1_05_f1)

    test_batches = list(tqdm(test_dataloader, disable=args.on_cluster))
    indices = [0, len(test_batches) // 2, len(test_batches) - 1]

    for i in indices:
        image_batch, labels_batch, _ = test_batches[i]

        image_batch = image_batch[0].unsqueeze(0).to(test_device)
        labels = labels_batch[0].unsqueeze(0).to(test_device)

        output = test_model(image_batch)[0].unsqueeze(0)

        current_save_str = f"{save_str}_tag{i}"

        if len(image_batch[0]) == 3:
            input1 = image_batch[0]
        else:
            input1 = image_batch[0][0]

        labels_dst = labels[0]
        pred = output.squeeze(0).detach().cpu().numpy()
        gradflows, cellprob = pred[:2], test_sigmoid(pred[-1])

        label, _ = compute_masks(gradflows, cellprob, use_gpu=True, device=test_device)
        lab = torch.from_numpy(label.astype(np.int32)).long().unsqueeze(0)

        if lab.squeeze().dim() == 2:
            show_images([input1] + [label_i for label_i in labels_dst] + [lab] + [out for out in output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label" for _ in labels_dst] + ["Prediction"] + ["Out" for _ in output[0]],
                        labels=[1, 2])
        else:
            show_images([input1] + [label_i for label_i in labels_dst] + [label_i for label_i in lab] + [out for out in
                                                                                                         output[0]],
                        save_str=current_save_str,
                        titles=["Source"] + ["Label: Nuclei", "Label: Cells"] + ["Prediction: Nuclei",
                                                                                 "Prediction: Cells"] + ["Out" for _ in
                                                                                                         output[0]],
                        labels=[1, 2, 3, 4], n_cols=5)

    end = time.time()
    return np.mean(test_loss), mean1_f1, end - start, mean1_05_f1



# import fastremap
class Segmentation_Dataset():
    def __init__(self, img, 
                label, 
                common_transforms=True,
                metadata=None, 
                size=(256, 256), 
                augmentation_dict=None,
                dim_in=3, 
                debug=False, 
                cells_and_nuclei=False, 
                target_segmentation="N", 
                channel_invariant = False,
                random_seed = None):
        
        self.X = img
        self.Y = label
        self.common_transforms = common_transforms

        assert len(self.X) == len(self.Y), "The number of images and labels must be the same"
        if len(metadata) == 0:
            self.metadata = [None] * len(self.X)
        else:
            self.metadata = metadata

        assert len(self.X) == len(self.metadata), print("The number of images and metadata must be the same")
        self.size = size
        self.Augmenter = Augmentations(augmentation_dict=augmentation_dict, 
                                       debug=debug, 
                                       shape=self.size,
                                       dim_in=dim_in, 
                                       cells_and_nuclei=cells_and_nuclei,
                                       target_segmentation=target_segmentation, 
                                       channel_invariant = channel_invariant,
                                       random_seed = random_seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        data = self.X[i]
        label = self.Y[i]
        meta = self.metadata[i]

        if self.common_transforms:
            data, label = self.Augmenter(data, label, meta)

        if len(label.shape) == 2:
            label = label[None, :]
        if len(data.shape) == 2:
            data = data[None, :]

        assert not data.isnan().any(), "Tranformed images contains NaN"
        assert not label.isnan().any(), "Transformed labels contains NaN"

        return data.float(), label



def plot_loss(_model):
    loss_fig = plt.figure()
    timer = loss_fig.canvas.new_timer(interval=300000)
    timer.add_callback(plt.close)

    losses = [param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None]
    names = [name for name, param in _model.named_parameters() if param.grad is not None]

    plt.plot(losses)
    plt.xticks(np.arange(len(names))[::1], names[::1])
    plt.xticks(fontsize=8, rotation=90)
    spacing = 0.5
    loss_fig.subplots_adjust(bottom=spacing)
    timer.start()
    plt.show()



def check_max_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.max()


def check_min_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.min()


def check_mean_grad(_model):
    losses = np.array([param.grad.norm().item() for name, param in _model.named_parameters() if param.grad is not None])
    return losses.mean()

def optimize_hyperparameters(model,postprocessing_fn,
                              data_loader = None, 
                              val_images = None, 
                              val_labels = None,
                              max_evals = 50, 
                              verbose = False, 
                              threshold = [0.5, 0.7, 0.9], 
                              show_progressbar = True, 
                              device = None):


    from cellembed.utils.metrics import _robust_average_precision
    from cellembed.utils.utils import _choose_device

    from hyperopt import fmin
    from hyperopt import hp
    from hyperopt import Trials
    from hyperopt import tpe
    import copy

    if device is None:
        device = _choose_device()

    bayes_trials = Trials()

    space = {  # instanseg
        'mask_threshold': hp.uniform('mask_threshold', 0.3, 0.7),
        'seed_threshold': hp.uniform('seed_threshold', 0.7, 1),
        #'overlap_threshold': hp.uniform('overlap_threshold', 0.1, 0.9),
        #'min_size': hp.uniform('min_size', 0, 30),
      #  'peak_distance': hp.uniform('peak_distance', 3, 10),
        'mean_threshold': hp.uniform('mean_threshold', 0.0, 0.5)} #the max could be increased, but may cuase the method not to converge for some reason.
    
    _model = model # copy.deepcopy(model)
    _model.eval()
    predictions = []

    with torch.no_grad():
        if data_loader is not None:
            for image_batch, labels_batch, _ in data_loader:
                    image_batch = image_batch.to(device)
                    output = _model(image_batch).cpu()
                    predictions.extend([pred,masks] for pred,masks in zip(output,labels_batch))


            def objective(params={}):
                pred_masks = []
                gt_masks = []
                for pred, masks in predictions:
                    lab = postprocessing_fn(pred.to(device), **params).cpu()
                    pred_masks.append(lab)
                    gt_masks.append(masks)

                mean_f1 = _robust_average_precision(torch.stack(gt_masks),torch.stack(pred_masks),threshold = threshold)

                if type(mean_f1) == list:
                    mean_f1 = np.nanmean(mean_f1)

                return 1 - mean_f1
        
        elif val_images is not None and val_labels is not None:
            from cellembed.utils.tiling import _instanseg_padding, _recover_padding
            def objective(params={}):
                pred_masks = []
                gt_masks = []
                #randomly shuffle val_images and val_labels

                np.random.seed(0)
                indexes = np.random.permutation(len(val_images))[:300]
                indexes.sort()

                for i in indexes:
                    imgs = val_images[i]
                    gt_mask = val_labels[i]
                    with torch.no_grad():
                        imgs = imgs.to(device)
                        imgs, pad = _instanseg_padding(imgs, min_dim = 32)
                        output = _model(imgs[None,])
                        output = _recover_padding(output, pad).squeeze(0)
                        lab = postprocessing_fn(output.to(device), **params).cpu()
                        pred_masks.append(lab)
                        gt_masks.append(gt_mask)

                mean_f1 = _robust_average_precision(gt_masks,pred_masks,threshold = threshold)

                if type(mean_f1) == list:
                    mean_f1 = np.nanmean(mean_f1)

                return 1 - mean_f1
        else:
            raise ValueError("Either data_loader or val_images and val_labels must be provided")

        print("Optimizing hyperparameters")
        # Optimize
        best = fmin(fn=objective, 
                    space=space, 
                    algo=tpe.suggest,
                    max_evals=max_evals, 
                    trials=bayes_trials, 
                    rstate=np.random.default_rng(0),
                    show_progressbar = show_progressbar)
    
    if verbose:
        print(best)
    return best



