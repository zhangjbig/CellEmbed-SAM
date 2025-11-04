from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from skimage import io, morphology
from scipy import ndimage
import fastremap
import zipfile
import skimage
import tifffile
import gc

from cellembed.utils.data_download import create_raw_datasets_dir, create_processed_datasets_dir
from cellembed.utils.utils import _move_channel_axis
from cellembed.utils.augmentations import Augmentations
from cellembed.utils.utils import display_colourized
from cellembed.utils.pytorch_utils import torch_fastremap

from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label

def build_tnbc_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "TNBC_NucleiSegmentation",
    subdir: str = "TNBC_and_Brain_dataset",
    out_name: str = "TNBC_2018_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    extract_zip: bool = False,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the TNBC 2018 segmentation dataset and save it as a .pth file.

    Parameters
    ----------
    project, dataset : str
        Names used by create_raw_datasets_dir for organizing datasets.
    subdir : str
        Subdirectory containing Slide_/GT_ pairs (inside dataset root).
    out_name : str
        Output .pth file name.
    raw_root : str
        Path to raw datasets root (sets INSTANSEG_RAW_DATASETS).
    dataset_root : str
        Path to processed dataset root (sets INSTANSEG_DATASET_PATH).
    extract_zip : bool
        Whether to extract {dataset}.zip if it exists.
    seed : int
        Random seed for dataset splitting.
    train_ratio, val_ratio, test_ratio : float
        Ratios for splitting dataset. Must sum to 1.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve() if raw_root else Path("../Raw_Datasets/").resolve()
    dataset_root = Path(dataset_root).resolve() if dataset_root else Path("../cellembed/datasets/").resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory & optionally extract ---
    tnbc_dir: Path = create_raw_datasets_dir(project, dataset)
    zip_file_path: Path = tnbc_dir / f"{dataset}.zip"
    if extract_zip and zip_file_path.exists():
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(tnbc_dir)

    file_path = tnbc_dir / subdir
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    # --- Scan subdirectories for Slide_/GT_ pairs and build items ---
    items = []
    folders = sorted([p for p in file_path.iterdir() if p.is_dir()])


    for folder in tqdm(folders, desc="Scanning"):
        if "Slide_" in str(folder):
            print(folder)
            for fname in sorted(os.listdir(folder)):
                if fname.endswith(".DS_Store"):
                    continue

                img_path = folder.parent / folder.name / fname
                mask_path =  folder.parent / folder.name.replace("Slide_", "GT_") / fname

                # print(1, img_path)
                # print(2, mask_path)
                if not (img_path.exists() and mask_path.exists()):
                    continue

                # Load image
                moved = io.imread(str(img_path))
                if _move_channel_axis(moved).shape[0] == 4:  # Remove alpha channel
                    moved = _move_channel_axis(moved)[:3]

                # Load mask -> connected components -> relabel
                masks = io.imread(str(mask_path))
                lab, n_labels = ndimage.label(masks > 0)
                labels, remapping = fastremap.renumber(skimage.morphology.label(lab), in_place=True)
                labels = fastremap.refit(labels)

                item = {
                    'cell_masks': labels,
                    'image': moved,
                    "specimen": "TNBC_2018",
                    "parent_dataset": "TNBC_2018",
                    'licence': "CC BY 4.0",
                    'pixel_size': 0.25,          # microns per pixel
                    'image_modality': "Brightfield",
                    'stain': "H&E",
                }
                items.append(item)

    if not items:
        raise RuntimeError("No valid Slide_/GT_ pairs found. Please check directory and naming.")

    # --- Split dataset randomly and save ---
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.")

    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        'Train': items[:n_train],
        'Validation': items[n_train:n_train + n_val],
        'Test': items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats

def build_nuinsseg_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "NuInsSeg",
    subdir: str = "NuInsSeg",
    out_name: str = "NuInsSeg_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the NuInsSeg segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    nuinsseg_dir: Path = create_raw_datasets_dir(project, dataset)
    file_path = nuinsseg_dir / subdir
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    items = []
    folders = sorted([p for p in file_path.iterdir() if p.is_dir()])

    for folder in tqdm(folders, desc="Scanning"):
        img_folder = folder / "tissue images"
        for fname in sorted(os.listdir(img_folder)):
            if ".DS_Store" in fname:
                continue
            img_path = img_folder / fname
            mask_path = img_path.with_name(fname.replace(".png", ".tif")).parent.parent / "label masks modify" / fname.replace(".png", ".tif")

            if not (img_path.exists() and mask_path.exists()):
                continue

            image = io.imread(str(img_path))[:, :, :3].astype(np.uint8)
            masks = io.imread(str(mask_path))

            labels, _ = fastremap.renumber(masks, in_place=True)
            labels = fastremap.refit(labels)

            item = {
                'cell_masks': labels,
                'image': image,
                "specimen": f"NuInsSeg_{folder.stem}",
                "parent_dataset": "NuInsSeg",
                'licence': "CC BY 4.0",
                'pixel_size': 0.25,
                'image_modality': "Brightfield",
                'stain': "H&E",
            }
            items.append(item)

    if not items:
        raise RuntimeError("No valid image/mask pairs found.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.")

    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        'Train': items[:n_train],
        'Validation': items[n_train:n_train + n_val],
        'Test': items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats


def build_ihc_tma_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "IHC_TMA_dataset",
    subdir: str = "IHC_TMA_dataset",
    out_name: str = "IHC_TMA_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the IHC_TMA segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """

    def separate_touching_objects(lab):
        mask = lab > 0
        distance_map = distance_transform_edt(mask)
        markers = distance_map > 2
        markers = label(markers)
        labels = watershed(-distance_map, markers, mask=mask)
        labels = labels + lab
        return labels

    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    ihc_tma_dir: Path = create_raw_datasets_dir(project, dataset)
    file_path = ihc_tma_dir / subdir / "images"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    items = []
    files = sorted(list(file_path.iterdir()))

    for file in tqdm(files, desc="Scanning"):
        if ".DS_Store" in file.name:
            continue

        image = io.imread(str(file))
        masks = np.load(str(file).replace("images", "masks").replace(".png", ".npy"))

        n_masks = np.max(masks[0:2], axis=0)
        masks = separate_touching_objects(n_masks)
        labels, _ = fastremap.renumber(masks, in_place=True)
        labels = fastremap.refit(labels)

        item = {
            'cell_masks': labels,
            'image': image,
            "specimen": "IHC_TMA",
            "parent_dataset": "IHC_TMA",
            'licence': "CC BY 4.0",
            'pixel_size': 0.25,
            'image_modality': "Brightfield",
            'stain': "IHC",
        }
        items.append(item)

    if not items:
        raise RuntimeError("No valid image/mask pairs found.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.")

    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        'Train': items[:n_train],
        'Validation': items[n_train:n_train + n_val],
        'Test': items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats


def build_lynsec_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "LyNSeC",
    subdir: str = "LyNSeC",
    out_name: str = "LyNSeC_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the LyNSeC segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    lynsec_dir: Path = create_raw_datasets_dir(project, dataset)
    file_path = lynsec_dir / subdir
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    items = []
    folders = sorted(list(file_path.iterdir()))

    for i, folder in enumerate(tqdm(folders, desc="Scanning")):
        if folder.stem == "lynsec 2":
            continue  # Skip incorrectly annotated folder

        for file in sorted(os.listdir(folder)):
            file_ = folder / file
            if ".DS_Store" in str(file_):
                continue

            data = np.load(str(file_))
            image = data[:, :, :3].astype(np.uint8)
            masks = data[:, :, 3].copy()
            labels, _ = fastremap.renumber(masks, in_place=True)
            labels = fastremap.refit(labels)

            item = {
                'cell_masks': labels,
                'image': image,
                "specimen": "LyNSeC",
                "parent_dataset": "LyNSeC",
                'licence': "CC BY 4.0",
                'pixel_size': 0.25,
                'image_modality': "Brightfield",
                'stain': "H&E" if i > 1 else "IHC",
            }
            items.append(item)

    if not items:
        raise RuntimeError("No valid items found.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.")

    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        'Train': items[:n_train],
        'Validation': items[n_train:n_train + n_val],
        'Test': items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats


def build_monuseg_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "MoNuSeg",
    subdir: str = "MoNuSeg/monuseg-2018/download",
    out_name: str = "MoNuSeg_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
):
    """
    Build the MoNuSeg 2018 segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    monuseg_dir: Path = create_raw_datasets_dir(project, dataset)
    file_path = monuseg_dir / subdir
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    Segmentation_Dataset = {"Train": [], "Validation": [], "Test": []}

    # --- Process Test Data ---
    test_items = []
    test_files = sorted(list((file_path / "test/images").iterdir()))
    for file in tqdm(test_files, desc="Test"):
        if ".DS_Store" in str(file):
            continue
        image = io.imread(str(file))[:, :, :3].astype(np.uint8)
        masks = io.imread(str(file).replace("images", "masks"))
        labels, _ = fastremap.renumber(masks, in_place=True)
        labels = fastremap.refit(labels)
        test_items.append({
            'cell_masks': labels,
            'image': image,
            "specimen": "MoNuSeg",
            "parent_dataset": "MoNuSeg",
            'licence': "CC BY NC 4.0",
            'pixel_size': 0.25,
            'image_modality': "Brightfield",
            'stain': "H&E",
        })
    Segmentation_Dataset['Test'] = test_items

    # --- Process Train Data ---
    train_items = []
    train_files = sorted(list((file_path / "train/images").iterdir()))
    for file in tqdm(train_files, desc="Train"):
        if ".DS_Store" in str(file):
            continue
        image = io.imread(str(file))[:, :, :3].astype(np.uint8)
        masks = io.imread(str(file).replace("images", "masks"))
        labels, _ = fastremap.renumber(masks, in_place=True)
        labels = fastremap.refit(labels)
        train_items.append({
            'cell_masks': labels,
            'image': image,
            "specimen": "MoNuSeg",
            "parent_dataset": "MoNuSeg",
            'licence': "CC BY NC 4.0",
            'pixel_size': 0.25,
            'image_modality': "Brightfield",
            'stain': "H&E",
        })

    rng = np.random.RandomState(seed)
    rng.shuffle(train_items)
    n_train = int(len(train_items) * train_ratio)
    Segmentation_Dataset['Train'] = train_items[:n_train]
    Segmentation_Dataset['Validation'] = train_items[n_train:]

    # --- Save ---
    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": len(train_items) + len(test_items),
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats

def build_bsst265_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "BSST265",
    subdir: str = "BSST265",
    out_name: str = "BSST265_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the BSST265 segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    bsst265_dir: Path = create_raw_datasets_dir(project, dataset)
    metadata = pd.read_csv(bsst265_dir / subdir / "image_description.csv", sep=";")
    file_path = bsst265_dir / subdir / "rawimages"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    items = []
    files = sorted(list(file_path.iterdir()))

    for file in tqdm(files, desc="Scanning"):
        metadata_row = metadata[metadata["Image_Name"] == file.stem]
        if metadata_row.empty:
            continue

        magnification = metadata_row["Magnification"].values[0]
        if magnification == "20x":
            pixel_size = 0.323
        elif magnification == "40x":
            pixel_size = 0.161
        elif magnification == "63x":
            pixel_size = 0.102
        else:
            pixel_size = 0.5

        img_file = str(file)
        mask_file = img_file.replace("rawimages", "groundtruth")

        image = io.imread(img_file)
        image = display_colourized(image)
        labels = io.imread(mask_file)

        _, area = np.unique(labels[labels > 0], return_counts=True)
        if len(area) == 0:
            print(f"Warning: Skipping empty label image: {file.name}")
            continue

        augmenter = Augmentations(shape=(2, 2))
        input_tensor, labels = augmenter.to_tensor(image, labels, normalize=False)

        current_pixel_size = 0.5 / ((np.median(area) ** 0.5) / (300 ** 0.5))
        tensor, labels = augmenter.torch_rescale(
            input_tensor,
            labels,
            current_pixel_size=current_pixel_size,
            requested_pixel_size=0.5,
            crop=False,
        )

        item = {
            'cell_masks': torch_fastremap(labels).squeeze(),
            'image': fastremap.refit(np.array(tensor.permute(1, 2, 0).byte())),
            "specimen": "BSST265",
            "parent_dataset": "BSST265",
            'licence': "CC0",
            'pixel_size': pixel_size,
            'nuclei_channels': [0],
        }
        items.append(item)

    if not items:
        raise RuntimeError("No valid items found.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.")

    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        'Train': items[:n_train],
        'Validation': items[n_train:n_train + n_val],
        'Test': items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats


def build_dsb_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "DSB",
    subdir: str = "stage1_train_instance_images",
    out_name: str = "DSB_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the DSB segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    dsb_dir: Path = create_raw_datasets_dir(project, dataset)
    file_path = dsb_dir / subdir
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    items = []
    train_files = sorted(list(file_path.iterdir()))

    for file in tqdm(train_files, desc="Scanning"):
        if ".DS_Store" in str(file):
            continue
        image = io.imread(str(file))[:, :, :3].astype(np.uint8)
        masks = io.imread(str(file).replace("images", "masks"))
        labels, _ = fastremap.renumber(masks, in_place=True)
        labels = fastremap.refit(labels)
        item = {
            'cell_masks': labels,
            'image': image,
            "specimen": "DSB",
            "parent_dataset": "DSB",
            'licence': "CC BY NC 4.0",
            'pixel_size': 0.25,
            'image_modality': "Brightfield",
            'stain': "H&E",
        }
        items.append(item)

    if not items:
        raise RuntimeError("No valid items found.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.")

    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        'Train': items[:n_train],
        'Validation': items[n_train:n_train + n_val],
        'Test': items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats

def build_deepbacs_dataset(
    project: str = "Cell_Segmentation",
    dataset: str = "DeepBacs",
    subdir: str = "",
    out_name: str = "DeepBacs_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
):
    """
    Build the DeepBacs segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    deepbacs_dir: Path = create_raw_datasets_dir(project, dataset)
    file_path = deepbacs_dir / subdir
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {file_path}")

    Segmentation_Dataset = {"Train": [], "Validation": [], "Test": []}

    # --- Process Test Data ---
    test_items = []
    test_files = sorted(list((file_path / "test/source").iterdir()))
    for file in tqdm(test_files, desc="Test"):
        if ".DS_Store" in str(file):
            continue
        data = io.imread(str(file))
        data = display_colourized(data)
        image = data[:, :, :3].astype(np.uint8)
        masks = io.imread(str(file).replace("source", "target"))
        labels, _ = fastremap.renumber(masks, in_place=True)
        labels = fastremap.refit(labels)
        test_items.append({
            'cell_masks': labels,
            'image': image,
            "specimen": "DeepBacs",
            "parent_dataset": "DeepBacs",
            'licence': "CC BY NC 4.0",
            'pixel_size': 0.25,
            'image_modality': "Brightfield",
            'stain': "H&E",
        })
    Segmentation_Dataset['Test'] = test_items

    # --- Process Train Data ---
    train_items = []
    train_files = sorted(list((file_path / "training/source").iterdir()))
    for file in tqdm(train_files, desc="Train"):
        if ".DS_Store" in str(file):
            continue
        data = io.imread(str(file))
        data = display_colourized(data)
        image = data[:, :, :3].astype(np.uint8)
        masks = io.imread(str(file).replace("source", "target"))
        labels, _ = fastremap.renumber(masks, in_place=True)
        labels = fastremap.refit(labels)
        train_items.append({
            'cell_masks': labels,
            'image': image,
            "specimen": "DeepBacs",
            "parent_dataset": "DeepBacs",
            'licence': "CC BY NC 4.0",
            'pixel_size': 0.25,
            'image_modality': "Brightfield",
            'stain': "H&E",
        })

    rng = np.random.RandomState(seed)
    rng.shuffle(train_items)
    n_train = int(len(train_items) * train_ratio)
    Segmentation_Dataset['Train'] = train_items[:n_train]
    Segmentation_Dataset['Validation'] = train_items[n_train:]

    # --- Save ---
    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": len(train_items) + len(test_items),
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats


def build_yeaz_dataset(
    project: str = "Cell_Segmentation",
    dataset: str = "YeaZ",
    subdirs: list = ["gold-standard-BF-V-1", "gold-standard-PhC-plus-2"],
    out_name: str = "YeaZ_dataset.pth",
    raw_root: str = "../Raw_Datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the YeaZ segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    yeza_dir: Path = create_raw_datasets_dir(project, dataset)
    train_folders = []
    for subdir in subdirs:
        train_dir = yeza_dir / subdir
        if not train_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {train_dir}")
        train_folders += sorted(list(train_dir.iterdir()))

    items = []
    for file in tqdm(train_folders, desc="Scanning"):
        if "mask" in file.name:
            continue

        image = io.imread(file)
        image = np.expand_dims(io.imread(file), axis=-1)
        image = display_colourized(image)

        labels = io.imread(str(file).replace("_im", "_mask"))
        if labels.ndim != 2:
            continue

        _, area = np.unique(labels[labels > 0], return_counts=True)
        if len(area) == 0:
            print(f"Warning: Skipping empty label image: {file.name}")
            continue

        augmenter = Augmentations(shape=(2, 2))
        input_tensor, labels = augmenter.to_tensor(image, labels, normalize=False)

        current_pixel_size = 0.5 / ((np.median(area) ** 0.5) / (300 ** 0.5))
        tensor, labels = augmenter.torch_rescale(
            input_tensor,
            labels,
            current_pixel_size=current_pixel_size,
            requested_pixel_size=0.5,
            crop=False,
        )

        item = {
            'cell_masks': torch_fastremap(labels).squeeze(),
            'image': fastremap.refit(np.array(tensor.permute(1, 2, 0).byte())),
            "specimen": "YeaZ",
            "parent_dataset": "YeaZ",
            'pixel_size': 0.5,
            'licence': "Non-Commercial",
            'image_modality': "Brightfield",
            'file_name': file.name,
            'original_size': image.shape,
        }
        items.append(item)

    if not items:
        raise RuntimeError("No valid items found.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.")

    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        'Train': items[:n_train],
        'Validation': items[n_train:n_train + n_val],
        'Test': items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats

def build_neurips_dataset(
    project: str = "Cell_Segmentation",
    dataset: str = "NeurIPS_CellSeg",
    out_name: str = "Neurips_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
):
    """
    Build the NeurIPS CellSeg dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    # --- Configure paths and environment variables ---
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    # --- Locate raw dataset directory ---
    neurips_dir: Path = create_raw_datasets_dir(project, dataset)
    file_path = Path(neurips_dir)

    Segmentation_Dataset = {"Train": [], "Validation": [], "Test": []}

    def process_split(split: str, is_train: bool = True):
        items = []
        image_dir = file_path / f"{split}/images"
        label_dir = file_path / f"{split}/labels"
        files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.bmp','.png', '.tif', '.tiff'] and not f.name.startswith('.')])

        for file in tqdm(files, desc=split):
            data = io.imread(str(file))
            if data.ndim == 2:
                data = display_colourized(data)
            image = data[:, :, :3].astype(np.uint8)

            stem = file.stem
            label_path = None
            for ext in ['.bmp','.png','.tif','.tiff']:
                candidate = label_dir / (stem + "_label" + ext)
                if candidate.exists():
                    label_path = candidate
                    break
            if label_path is None:
                print(f"[Warning] Label not found for {file.name}, skipped.")
                continue

            masks = io.imread(str(label_path))
            labels, _ = fastremap.renumber(masks, in_place=True)

            _, area = np.unique(labels[labels > 0], return_counts=True)
            if len(area) == 0:
                print(f"Warning: Skipping empty label image: {file.name}")
                continue

            augmenter = Augmentations(shape=(2, 2))
            tensor, labels = augmenter.to_tensor(image, labels, normalize=False)

            item = {
                'cell_masks': torch_fastremap(labels).squeeze(),
                'image': fastremap.refit(np.array(tensor.permute(1, 2, 0).byte())),
                "specimen": "Neurips",
                "parent_dataset": "Neurips",
                'licence': "CC BY NC 4.0",
                'pixel_size': 0.25,
                'image_modality': "Brightfield",
                'stain': "H&E",
            }
            items.append(item)
        return items

    # Process Test
    Segmentation_Dataset['Test'] = process_split("Tuning", is_train=False)

    # Process Train
    train_items = process_split("Training", is_train=True)
    rng = np.random.RandomState(seed)
    rng.shuffle(train_items)
    n_train = int(len(train_items) * train_ratio)
    Segmentation_Dataset['Train'] = train_items[:n_train]
    Segmentation_Dataset['Validation'] = train_items[n_train:]

    # --- Save ---
    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": len(train_items) + len(Segmentation_Dataset['Test']),
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats

def build_tissuenet_dataset(
    project: str = "Cell_Segmentation",
    dataset: str = "TissueNet",
    version: str = "tissuenet_v1.1",
    out_name: str = "TissueNet_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
):
    """
    Build the TissueNet segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    tissuenet_dir = create_raw_datasets_dir(project, dataset)
    processed_tissuenet_dir = create_processed_datasets_dir("tissuenet_data_processed")

    def get_data(split: str):
        out_path = processed_tissuenet_dir / split
        out_path.mkdir(parents=True, exist_ok=True)

        file_path = tissuenet_dir / f"{version}/{version}_{split}.npz"
        data = np.load(file_path, allow_pickle=True, mmap_mode='r')

        items = []
        imgs = data["X"]
        labels = data["y"]
        metas = data["meta"]

        for i in tqdm(range(len(imgs)), desc=f"{split}"):
            image = display_colourized(imgs[i])
            label = labels[i]
            meta = metas[i + 1]

            tifffile.imwrite(out_path / f"image_{i}.tif", fastremap.refit(image))
            tifffile.imwrite(out_path / f"cell_masks_{i}.tif", fastremap.refit(fastremap.renumber(label[:, :, 0])[0]))
            tifffile.imwrite(out_path / f"nucleus_masks_{i}.tif", fastremap.refit(fastremap.renumber(label[:, :, 1])[0]))

            relative_path_img = os.path.relpath(str(out_path / f"image_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_cell = os.path.relpath(str(out_path / f"cell_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_nucleus = os.path.relpath(str(out_path / f"nucleus_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])

            item = {
                'image': relative_path_img,
                'cell_masks': relative_path_cell,
                'nucleus_masks': relative_path_nucleus,
                "parent_dataset": "TissueNet",
                'specimen': f"TissueNet_{meta[5]}",
                'image_modality': "Fluorescence",
                'pixel_size': meta[2],
                'nuclei_channels': [0],
            }
            items.append(item)

        data.close()
        del data, imgs, labels, metas
        gc.collect()

        return items

    Segmentation_Dataset = {
        'Train': get_data("train"),
        'Validation': get_data("val"),
        'Test': get_data("test"),
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": len(Segmentation_Dataset['Train']) + len(Segmentation_Dataset['Validation']) + len(Segmentation_Dataset['Test']),
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats


def build_pannuke_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "PanNuke",
    out_name: str = "PanNuke_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
):
    """
    Build the PanNuke segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    pannuke_dir = create_raw_datasets_dir(project, dataset)
    processed_pannuke_dir = create_processed_datasets_dir("pannuke_data")

    def get_data(split: str):
        if split == "train":
            fold = "1"
        elif split == "val":
            fold = "2"
        elif split == "test":
            fold = "3"
        else:
            raise ValueError("split must be one of train/val/test")

        out_path = create_processed_datasets_dir("pannuke_data", split)

        masks = np.load(pannuke_dir / f"Fold {fold}/masks/fold{fold}/masks.npy", allow_pickle=True).astype(np.int16)
        images = np.load(pannuke_dir / f"Fold {fold}/images/fold{fold}/images.npy", allow_pickle=True).astype(np.uint8)
        types = np.load(pannuke_dir / f"Fold {fold}/images/fold{fold}/types.npy", allow_pickle=True)

        items = []
        for i in tqdm(range(len(images)), desc=f"{split}"):
            assert (np.unique(masks[i, :, :, :-1]) == np.unique(masks[i, :, :, :-1].max(-1))).all()
            label = masks[i, :, :, :-1].max(-1)
            label, _ = fastremap.renumber(label, in_place=True)
            classes = (np.argmax(masks[i, :, :, :-1], axis=-1) + 1) * (label > 0).astype(np.int8)

            image = images[i]

            tifffile.imwrite(out_path / f"image_{i}.tif", image)
            tifffile.imwrite(out_path / f"nucleus_masks_{i}.tif", fastremap.refit(fastremap.renumber(label)[0]))
            tifffile.imwrite(out_path / f"class_masks_{i}.tif", classes)

            relative_path_img = os.path.relpath(str(out_path / f"image_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_nucleus = os.path.relpath(str(out_path / f"nucleus_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
            relative_path_class = os.path.relpath(str(out_path / f"class_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])

            item = {
                'image': relative_path_img,
                'cell_masks': relative_path_nucleus,
                'class_masks': relative_path_class,
                "parent_dataset": "PanNuke",
                'licence': "Attribution-NonCommercial-ShareAlike 4.0 International",
                'image_modality': "Brightfield",
                'pixel_size': 0.25,
                'specimen': f"PanNuke_{types[i]}",
            }
            items.append(item)
        return items

    Segmentation_Dataset = {
        'Train': get_data("train"),
        'Validation': get_data("val"),
        'Test': get_data("test"),
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": len(Segmentation_Dataset['Train']) + len(Segmentation_Dataset['Validation']) + len(Segmentation_Dataset['Test']),
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats

def build_conic_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "CoNic",
    subdir: str = "data",
    out_name: str = "CoNic_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the CoNic segmentation dataset and save it as a .pth file.

    Returns
    -------
    saved_path : Path
        Path to the saved .pth file.
    stats : dict
        Statistics including total, train, validation, test counts, and save path.
    """
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    os.environ['INSTANSEG_RAW_DATASETS'] = str(raw_root)
    os.environ['INSTANSEG_DATASET_PATH'] = str(dataset_root)

    conic_dir = create_raw_datasets_dir(project, dataset)
    out_path = create_processed_datasets_dir("conic_data", subdir)

    masks = np.load(Path(conic_dir) / subdir / "labels.npy", allow_pickle=True).astype(np.int16)
    images = np.load(Path(conic_dir) / subdir / "images.npy", allow_pickle=True).astype(np.uint8)
    patch_info = pd.read_csv(Path(conic_dir) / subdir / "patch_info.csv", skiprows=1, header=None)

    items = []
    for i in tqdm(range(len(images)), desc="Scanning"):
        image = images[i]
        label = masks[i]

        tifffile.imwrite(out_path / f"image_{i}.tif", fastremap.refit(image))
        tifffile.imwrite(out_path / f"class_masks_{i}.tif", fastremap.refit(fastremap.renumber(label[:, :, 1])[0]))
        tifffile.imwrite(out_path / f"nucleus_masks_{i}.tif", fastremap.refit(fastremap.renumber(label[:, :, 0])[0]))

        relative_path_img = os.path.relpath(str(out_path / f"image_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
        relative_path_class = os.path.relpath(str(out_path / f"class_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])
        relative_path_nucleus = os.path.relpath(str(out_path / f"nucleus_masks_{i}.tif"), os.environ['INSTANSEG_DATASET_PATH'])

        item = {
            'image': relative_path_img,
            'class_masks': relative_path_class,
            'cell_masks': relative_path_nucleus,
            "parent_dataset": "CoNic",
            'licence': "CC BY NC 4.0",
            'pixel_size': 0.25,
            'image_modality': "Brightfield",
            'stain': "H&E",
            'specimen': f"CoNic_{patch_info.iloc[i,0].split('_')[0]}",
        }
        items.append(item)

    if not items:
        raise RuntimeError("No valid items found.")

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.")

    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        'Train': items[:n_train],
        'Validation': items[n_train:n_train + n_val],
        'Test': items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset['Train']),
        "Validation": len(Segmentation_Dataset['Validation']),
        "Test": len(Segmentation_Dataset['Test']),
        "saved_to": str(saved_path),
    }
    return saved_path, stats




def build_human_rna_dataset(
    dataset: str,
    specimen: str,
    out_name: str,
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build a Human RNA dataset (Cortex, Tonsil, Liver, Pancreas).
    """
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = create_raw_datasets_dir("Cell_Segmentation", dataset)
    files = sorted(list((dataset_dir / "CellComposite_Processed").iterdir()))

    items = []
    for file in tqdm(files, desc=f"Processing {dataset}"):
        data = io.imread(str(file))
        if data.ndim != 3 or data.shape[2] != 3:
            continue
        image = data[:, :, :3].astype(np.uint8)

        label_file = file.name.replace("CellComposite_Processed", "CellLabels_Processed")
        label_path = file.parent.parent / "CellLabels_Processed" / label_file
        label_path = label_path.with_suffix(".tif")

        if not label_path.exists():
            continue

        masks = io.imread(str(label_path))
        masks, _ = fastremap.renumber(masks, in_place=True)

        augmenter = Augmentations(shape=(2, 2))
        input_tensor, labels = augmenter.to_tensor(image, masks, normalize=False)

        item = {
            "cell_masks": np.array(torch_fastremap(labels).squeeze()),
            "image": fastremap.refit(np.array(input_tensor.byte())),
            "parent_dataset": "Human_RNA",
            "specimen": specimen,
            "licence": "CC BY NC 4.0",
            "pixel_size": 0.5,
            "image_modality": "Fluorescence",
        }
        items.append(item)

    rng = np.random.RandomState(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        "Train": items[:n_train],
        "Validation": items[n_train:n_train + n_val],
        "Test": items[n_train + n_val:],
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset["Train"]),
        "Validation": len(Segmentation_Dataset["Validation"]),
        "Test": len(Segmentation_Dataset["Test"]),
        "saved_to": str(saved_path),
    }
    return saved_path, stats



def build_cellbindb_dataset(
    project: str = "Nucleus_Segmentation",
    dataset: str = "CellBinDB",
    out_name: str = "CellBinDB_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the CellBinDB segmentation dataset and save it as a .pth file.
    """
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    cellbindb_dir = create_raw_datasets_dir(project, dataset)
    dataset_path = sorted((cellbindb_dir / "CellBinDB").iterdir())

    items = []
    for tissue_dir in tqdm(dataset_path, desc="Scanning tissues"):
        if not tissue_dir.is_dir():
            continue
        for sample_dir in sorted(tissue_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            img_file = sample_dir / f"{sample_dir.name}-img.tif"
            instance_mask_file = sample_dir / f"{sample_dir.name}-instancemask.tif"

            if img_file.exists() and instance_mask_file.exists():
                image = io.imread(img_file)
                instance_mask = io.imread(instance_mask_file)

                item = {
                    "cell_masks": instance_mask,
                    "image": image,
                    "parent_dataset": "CellBinDB",
                    "specimen": "CellBinDB",
                    "licence": "CC BY 4.0",
                    "pixel_size": 0.5,  # update if ground truth available
                }
                items.append(item)

    if not items:
        raise RuntimeError("No valid items found in CellBinDB.")

    # shuffle and split
    rng = np.random.RandomState(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        "Train": items[:n_train],
        "Validation": items[n_train:n_train + n_val],
        "Test": items[n_train + n_val:]
    }

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": n,
        "Train": len(Segmentation_Dataset["Train"]),
        "Validation": len(Segmentation_Dataset["Validation"]),
        "Test": len(Segmentation_Dataset["Test"]),
        "saved_to": str(saved_path),
    }
    return saved_path, stats

def build_cellpose_dataset(
    project: str = "Cell_Segmentation",
    dataset: str = "Cellpose",
    out_name: str = "Cellpose_dataset.pth",
    raw_root: str = "../raw datasets/",
    dataset_root: str = "../cellembed/datasets/",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    Build the Cellpose segmentation dataset and save it as a .pth file.
    """
    raw_root = Path(raw_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    cellpose_dir = create_raw_datasets_dir(project, dataset)

    def process_files(file_path):
        items = []
        files = sorted(list(file_path.iterdir()))
        for file in tqdm(files, desc=f"Processing {file_path.name}"):
            if "masks" in file.name:
                continue
            image = io.imread(file)
            labels = io.imread(str(file).replace("_img", "_masks"))
            _, area = np.unique(labels[labels > 0], return_counts=True)
            if len(area) == 0:
                continue

            augmenter = Augmentations(shape=(2, 2))
            input_tensor, labels = augmenter.to_tensor(image, labels, normalize=False)
            tensor, labels = augmenter.torch_rescale(
                input_tensor,
                labels,
                current_pixel_size=0.5 / ((np.median(area) ** 0.5) / (300 ** 0.5)),
                requested_pixel_size=0.5,
                crop=False
            )

            item = {
                "cell_masks": torch_fastremap(labels).squeeze(),
                "image": fastremap.refit(np.array(tensor.byte())),
                "parent_dataset": "Cellpose",
                "specimen": "Cellpose",
                "licence": "Non-Commercial",
                "image_modality": "Fluorescence",
                "file_name": file.name,
                "original_size": image.shape,
            }
            items.append(item)
        return items

    # Process Train
    train_items = process_files(Path(cellpose_dir) / "train")

    rng = np.random.RandomState(seed)
    rng.shuffle(train_items)

    n = len(train_items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    Segmentation_Dataset = {
        "Train": train_items[:n_train],
        "Validation": train_items[n_train:n_train + n_val],
        "Test": train_items[n_train + n_val:]
    }

    # Process Test
    test_items = process_files(Path(cellpose_dir) / "test")
    Segmentation_Dataset["Test"].extend(test_items)

    saved_path = dataset_root / out_name
    torch.save(Segmentation_Dataset, str(saved_path))

    stats = {
        "total": len(train_items) + len(test_items),
        "Train": len(Segmentation_Dataset["Train"]),
        "Validation": len(Segmentation_Dataset["Validation"]),
        "Test": len(Segmentation_Dataset["Test"]),
        "saved_to": str(saved_path),
    }
    return saved_path, stats


def build_all_datasets(dataset_root="../cellembed/datasets/", out_name="All_Datasets.pth"):
    dataset_root = Path(dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    All_Dataset = {"Train": [], "Validation": [], "Test": []}

    # dataset builders
    builders = [
        build_nuinsseg_dataset,
        build_ihc_tma_dataset,
        build_lynsec_dataset,
        build_tnbc_dataset,
        build_monuseg_dataset,
        build_bsst265_dataset,
        build_dsb_dataset,
        build_deepbacs_dataset,
        build_yeaz_dataset,
        build_neurips_dataset,
        build_tissuenet_dataset,
        build_pannuke_dataset,
        build_conic_dataset,
    ]

    for build_fn in builders:
        _, stats = build_fn()
        seg = torch.load(stats["saved_to"])
        All_Dataset["Train"].extend(seg["Train"])
        All_Dataset["Validation"].extend(seg["Validation"])
        All_Dataset["Test"].extend(seg["Test"])
        print(f"Loaded {build_fn.__name__}: {stats}")

    # human RNA datasets
    human_builders = [
        ("Human_Frontal_Cortex_Processed_clip", "Human_Cortex", "Human_Cortex_dataset.pth"),
        ("Human_Tonsil_Processed_clip", "Human_Tonsil", "Human_Tonsil_dataset.pth"),
        ("Human_Liver_Processed_clip", "Human_Liver", "Human_Liver_dataset.pth"),
        ("Human_Pancreas_Processed_clip", "Human_Pancreas", "Human_Pancreas_dataset.pth"),
    ]

    for dataset, specimen, out_name_single in human_builders:
        _, stats = build_human_rna_dataset(dataset, specimen, out_name_single)
        seg = torch.load(stats["saved_to"])
        All_Dataset["Train"].extend(seg["Train"])
        All_Dataset["Validation"].extend(seg["Validation"])
        All_Dataset["Test"].extend(seg["Test"])
        print(f"Loaded Human RNA dataset: {specimen}, {stats}")

    saved_path = dataset_root / out_name
    torch.save(All_Dataset, str(saved_path))

    stats = {
        "total": len(All_Dataset["Train"]) + len(All_Dataset["Validation"]) + len(All_Dataset["Test"]),
        "Train": len(All_Dataset["Train"]),
        "Validation": len(All_Dataset["Validation"]),
        "Test": len(All_Dataset["Test"]),
        "saved_to": str(saved_path),
    }
    return saved_path, stats


if __name__ == "__main__":

    saved_path, stats = build_tnbc_dataset()

    # saved_path, stats = build_all_datasets()
    print("Final merged dataset stats:", stats)

    # saved_path, stats = build_cellbindb_dataset()
    # saved_path, stats = build_cellpose_dataset()