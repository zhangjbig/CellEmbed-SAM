import sys
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
import cv2
import gc

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox, QGraphicsDropShadowEffect,
    QLineEdit, QAbstractItemView, QSizePolicy, QGridLayout, QSlider
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QFont, QFontDatabase
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal

if getattr(sys, 'frozen', False):
    os.environ["PYINSTALLER_TEMP"] = r"D:\pycache"

try:
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
    else:
        gpu_info = "No GPU detected"
except ImportError:
    gpu_info = "PyTorch not installed"

from cellembed.utils.model_loader import load_model
from cellembed.utils.augmentations import Augmentations
from cellembed.utils.tiling import _instanseg_padding, _recover_padding
from cellembed.utils.loss.cellembed_loss import InstanSeg
import colorcet as cc
from cellembed.utils.utils import label_to_color_image
from cellembed.scripts.test import sliding_window_center_inference


def segment_single_image(image_path, model, method, device, parser_args, model_dict, params=None, batch_size: int = 1):
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    Aug = Augmentations(dim_in=model_dict['dim_in'], shape=None)
    tensor_img, _ = Aug.to_tensor(img_np, None, normalize=True)
    tensor_img, _ = Aug.normalize(tensor_img)
    tensor_img = tensor_img.to(device)

    H, W = tensor_img.shape[1], tensor_img.shape[2]

    if H > 256 or W > 256:
        with torch.no_grad():
            pred = sliding_window_center_inference(
                tensor_img, model, device,
                patch_size=256, center_crop=128,
                padding=128, batch_size=batch_size,
                out_channels=method.dim_out
            ).to(device)

        if params is not None:
            with autocast():
                lab = method.postprocessing_seed(pred, **params, window_size=parser_args.window_size)
        else:
            with autocast():
                lab = method.postprocessing_seed(pred, img=tensor_img, window_size=parser_args.window_size)

    else:
        tensor_img, pad = _instanseg_padding(tensor_img, extra_pad=0, min_dim=32)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                pred = model(tensor_img[None, ])
        pred = _recover_padding(pred, pad).squeeze(0)

        if params is not None:
            with torch.amp.autocast("cuda"):
                lab = method.postprocessing_seed(pred, **params, window_size=parser_args.window_size)
        else:
            with torch.amp.autocast("cuda"):
                lab = method.postprocessing_seed(pred, img=tensor_img, window_size=parser_args.window_size)

    lab = lab.cpu().numpy().astype(np.int16)
    lab_ids = lab[0]

    colors = cc.cm.glasbey_bw_minc_20_minl_30_r.colors
    lab_color = label_to_color_image(lab_ids, colors=colors)

    return lab_color, lab_ids


class FolderSegThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, file_paths, output_dir, model, method, device, parser_args, model_dict, params, batch_size, alpha):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self.model = model
        self.method = method
        self.device = device
        self.parser_args = parser_args
        self.model_dict = model_dict
        self.params = params
        self.batch_size = batch_size
        self.alpha = alpha

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            from cellembed.utils.cellmorphology import CellImageAnalyzer
            from PIL import Image as PILImage

            exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            paths = [p for p in self.file_paths if os.path.splitext(p)[1].lower() in exts]
            total = len(paths)
            if total == 0:
                self.finished.emit("⚠ No images found in folder.")
                return

            for idx, path in enumerate(paths, 1):
                base = os.path.splitext(os.path.basename(path))[0]
                self.progress.emit(f"[{idx}/{total}] Segmenting: {base}")

                lab_color, lab_ids = segment_single_image(
                    path, self.model, self.method, self.device,
                    self.parser_args, self.model_dict, self.params,
                    batch_size=self.batch_size
                )

                unique_ids = np.unique(lab_ids)
                unique_ids = unique_ids[unique_ids > 0]
                id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
                new_lab = np.zeros_like(lab_ids, dtype=np.int32)
                for old_id, new_id in id_map.items():
                    new_lab[lab_ids == old_id] = new_id
                lab_ids = new_lab

                analyzer = CellImageAnalyzer(image_array=lab_ids.astype(np.uint16))
                analyzer.compute_contours_and_stats()
                analyzer.calculate_morphology_stats()
                df = analyzer.generate_dataframe()
                stats_path = os.path.join(self.output_dir, f"{base}_stats.csv")
                df.to_csv(stats_path, index=True)

                seg_img = PILImage.fromarray(lab_color.astype(np.uint8))
                seg_path = os.path.join(self.output_dir, f"{base}_seg.png")
                seg_img.save(seg_path)

                pil_img = PILImage.open(path).convert("RGB")
                pil_img = pil_img.resize((lab_color.shape[1], lab_color.shape[0]))
                img_np = np.array(pil_img).astype(np.float32) / 255.0
                lab_color_f = lab_color.astype(np.float32) / 255.0
                alpha = float(self.alpha)
                mask = np.any(lab_color_f > 0, axis=-1, keepdims=True)
                overlay_np = (1 - alpha) * img_np + alpha * lab_color_f
                overlay_np = np.where(mask, overlay_np, img_np)
                overlay_np = (overlay_np * 255).astype(np.uint8)
                overlay_path = os.path.join(self.output_dir, f"{base}_overlay.png")
                PILImage.fromarray(overlay_np).save(overlay_path)

            self.finished.emit(f"✅ Done. Saved to: {self.output_dir}")
        except Exception as e:
            self.finished.emit(f"❌ Failed: {e}")


class SegmentationThread(QThread):
    finished = pyqtSignal(object, object)  # lab_color, lab_ids

    def __init__(self, image_path, model, method, device, parser_args, model_dict, params, batch_size):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.method = method
        self.device = device
        self.parser_args = parser_args
        self.model_dict = model_dict
        self.params = params
        self.batch_size = batch_size

    def run(self):
        lab_color, lab_ids = segment_single_image(
            self.image_path,
            self.model,
            self.method,
            self.device,
            self.parser_args,
            self.model_dict,
            self.params,
            batch_size=self.batch_size,
        )
        self.finished.emit(lab_color, lab_ids)


class CellSegmentationGUI(QWidget):
    def __init__(self, model, method, device, parser_args, model_dict, params=None):
        super().__init__()

        self.PREVIEW = 400

        self.setWindowTitle("Cell Segmentation GUI")
        self.setGeometry(200, 200, 1600, 700)
        # self.setStyleSheet("background-color: #f8f9fa;")

        self.gpu_label = QLabel(f"GPU Info: {gpu_info}")
        self.gpu_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.gpu_label.setStyleSheet(
            "font-size: 12px; color: #444; padding: 4px 8px; background: #e9ecef; border-radius: 5px;"
        )
        self.gpu_label.setFixedHeight(28)

        self.gpu_mem_label = QLabel("VRAM: —")
        self.gpu_mem_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.gpu_mem_label.setStyleSheet(
            "font-size: 12px; color: #444; padding: 4px 8px; background: #f1f3f5; border-radius: 5px;"
        )
        self.gpu_mem_label.setFixedHeight(28)

        self.batch_size_label = QLabel("Tiles per batch:")
        self.batch_size_label.setStyleSheet("font-size: 12px; color: #495057;")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(1)
        self.batch_size_spin.setSingleStep(1)
        self.batch_size_spin.setFixedWidth(80)
        self.batch_size_spin.setStyleSheet("""
            QSpinBox {
                background: white; border: 1px solid #adb5bd; border-radius: 4px;
                padding: 2px; font-size: 12px; color: #343a40;
            }
        """)

        self.input_title = QLabel("📷 Input Image")
        self.input_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")

        self.input_label = QLabel()
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_label.setFixedSize(self.PREVIEW, self.PREVIEW)
        self.input_label.setStyleSheet(
            "border: 2px dashed #adb5bd; background: white; font-size: 13px; color: #6c757d;"
        )

        self.input_res_label = QLabel("", self.input_label)
        self.input_res_label.setStyleSheet(
            "font-size: 11px; color: white; background-color: rgba(0,0,0,128); padding: 2px;")
        self.input_res_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)

        effect = QGraphicsDropShadowEffect(self.input_res_label)
        effect.setBlurRadius(2)
        effect.setOffset(0, 0)
        effect.setColor(Qt.GlobalColor.black)
        self.input_res_label.setGraphicsEffect(effect)

        self.input_res_label.setGeometry(0, self.input_label.height() - 20, 100, 20)

        self.input_info_label = QLabel(self.input_label)
        self.input_info_label.setText(
            "<b>An overview of microscopy images</b><br>"
            "“29,566 segmented microscopic images, with 16,800 for training and augmentation via sampling.”"
        )
        self.input_info_label.setWordWrap(True)
        self.input_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_info_label.setStyleSheet(
            "font-size: 12px; color: white; background-color: rgba(0,0,0,160);"
            "padding: 6px 8px; border-radius: 8px;"
        )

        self.input_info_label.setFixedWidth(min(self.PREVIEW - 40, 320))

        self.input_info_label.show()
        self.input_info_label.raise_()
        self._center_input_info()

        self.input_btn = QPushButton("Import Image")
        self.input_btn.clicked.connect(self.load_input_image)

        self.segment_folder_btn = QPushButton("Segment Folder")
        self.segment_folder_btn.clicked.connect(self.segment_folder)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.input_btn)
        btn_row.addWidget(self.segment_folder_btn)
        btn_row.addWidget(self.clear_btn)

        input_layout = QVBoxLayout()
        input_layout.addWidget(self.input_title)
        input_layout.addWidget(self.input_label, alignment=Qt.AlignmentFlag.AlignCenter)
        input_layout.addSpacing(10)
        input_layout.addLayout(btn_row)
        input_layout.addStretch()

        self.output_title = QLabel("🧠 Segmented Output")
        self.output_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")

        self.output_label = QLabel()
        self.output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_label.setFixedSize(self.PREVIEW, self.PREVIEW)
        self.output_label.setStyleSheet(
            "border: 2px solid #adb5bd; background: white; font-size: 13px; color: #6c757d;"
        )
        self.output_label.setMouseTracking(True)
        self.output_label.installEventFilter(self)

        self.output_info_label = QLabel(self.output_label)
        self.output_info_label.setText(
            "<b>Image Encoder of SAM Weights (Large)</b><br>"
            "“Integrating a deep network for efficient cell centroid seed selection, improving accuracy and speeding up post-processing.”"
        )
        self.output_info_label.setWordWrap(True)
        self.output_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_info_label.setStyleSheet(
            "font-size: 12px; color: white; background-color: rgba(0,0,0,160);"
            "padding: 6px 8px; border-radius: 8px;"
        )
        self.output_info_label.setFixedWidth(min(self.PREVIEW - 40, 320))
        self.output_info_label.show()
        self.output_info_label.raise_()
        self._center_output_info()


        self.segment_btn = QPushButton("Segment Image")
        self.segment_btn.clicked.connect(self.run_segmentation)
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_output_image)

        self.outline_btn = QPushButton("Toggle Outline Mode")
        self.outline_btn.clicked.connect(self.toggle_outline_mode)

        for btn in [self.input_btn, self.segment_btn]:
            btn.setFixedWidth(150)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #007bff;
                    color: white;
                    border-radius: 6px;
                    padding: 6px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
            """)

        self.segment_folder_btn.setFixedWidth(150)
        self.segment_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #fd7e14;
                color: white;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e86d0a;
            }
            QPushButton:disabled {
                background-color: #f1b383; 
                color: #f8f9fa;
            }
        """)

        self.outline_btn.setFixedWidth(150)
        self.outline_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:disabled {
                background-color: #adb5bd;
                color: #f8f9fa;
            }
        """)

        self.save_btn.setFixedWidth(150)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #117a8b;
            }
        """)

        self.clear_btn.setFixedWidth(150)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #a71d2a; 
            }
        """)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.segment_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.outline_btn)
        btn_row.addStretch()

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_title)
        output_layout.addWidget(self.output_label, alignment=Qt.AlignmentFlag.AlignCenter)
        output_layout.addSpacing(10)
        output_layout.addLayout(btn_row)
        output_layout.addStretch()

        self.overlay_title = QLabel("🖼️ Overlay (Image + Labels)")
        self.overlay_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #495057;")

        self.overlay_label = QLabel()
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_label.setFixedSize(self.PREVIEW, self.PREVIEW)
        self.overlay_label.setStyleSheet(
            "border: 2px solid #adb5bd; background: white; font-size: 13px; color: #6c757d;"
        )

        overlay_layout = QVBoxLayout()
        overlay_layout.addWidget(self.overlay_title)
        overlay_layout.addWidget(self.overlay_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.setFixedWidth(150)
        self.alpha_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                height: 6px;
                background: #e0e0e0;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                border: 1px solid #5c5c5c;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """)
        self.alpha_slider.valueChanged.connect(self.update_overlay_alpha)

        self.save_overlay_btn = QPushButton("Save Overlay")
        self.save_overlay_btn.setFixedWidth(150)
        self.save_overlay_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #117a8b;
            }
        """)
        self.save_overlay_btn.clicked.connect(self.save_overlay_image)

        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.setMinimumWidth(100)
        self.export_csv_btn.setProperty("variant", "secondary")

        self.export_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.export_csv_btn.clicked.connect(self.export_csv)

        self.alpha_label = QLabel("Overlay Transparency:")
        self.alpha_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alpha_label.setStyleSheet("font-size: 12px; color: #495057;")

        Policy = getattr(QSizePolicy, "Policy", QSizePolicy)
        self.alpha_slider.setSizePolicy(Policy.Expanding, Policy.Fixed)

        overlay_controls = QHBoxLayout()
        overlay_controls.addWidget(self.alpha_label)
        overlay_controls.addWidget(self.alpha_slider, 1)
        overlay_controls.addStretch()
        overlay_controls.addWidget(self.save_overlay_btn)

        overlay_layout.addSpacing(10)
        overlay_layout.addLayout(overlay_controls)

        # --- Table and related widgets ---
        self.table_title = QLabel("Cell Morphology Table")
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["🔢 ID", "⬛ Area", "🧩 Compactness", "⭕ Circularity", "🧱 Solidity", "🥏 Eccentricity"]
        )

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.table.setStyleSheet("""
            QTableWidget {
                font-size: 12px;
                background: #ffffff;
                gridline-color: #adb5bd;
                border: 1px solid #adb5bd;
            }
            QHeaderView::section {
                background-color: #dee2e6;
                font-weight: bold;
                border: 1px solid #adb5bd;
            }
        """)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Enter cell ID and press Enter")
        self.search_edit.setFixedWidth(140)
        self.search_edit.returnPressed.connect(self.find_cell_id)

        self.search_btn = QPushButton("Find")
        self.search_btn.setProperty("variant", "secondary")
        self.search_btn.clicked.connect(self.find_cell_id)
        self.search_btn.setText("🔍 Find")

        self.search_clear_btn = QPushButton("Clear")
        self.search_clear_btn.setProperty("variant", "secondary")
        self.search_clear_btn.clicked.connect(self.clear_search_highlight)

        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        search_row = QHBoxLayout()
        search_row.addStretch()
        search_row.addWidget(QLabel("Find ID:"))
        search_row.addWidget(self.search_edit)
        search_row.addWidget(self.search_btn)
        search_row.addWidget(self.search_clear_btn)
        search_row.addStretch()

        self.table.cellClicked.connect(self.on_table_selection_changed)

        left_tools = QHBoxLayout()
        left_tools.addWidget(QLabel("Find ID:"))
        left_tools.addWidget(self.search_edit)
        left_tools.addWidget(self.search_btn)
        left_tools.addWidget(self.search_clear_btn)

        self.table.horizontalHeader().setSectionsMovable(False)
        self.table.verticalHeader().setSectionsMovable(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # --- New: columns as QWidget containers (kept), but no FixedWidth ---
        input_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setContentsMargins(0, 0, 0, 0)

        input_col = QWidget();  input_col.setLayout(input_layout)
        output_col = QWidget(); output_col.setLayout(output_layout)
        overlay_col = QWidget(); overlay_col.setLayout(overlay_layout)

        input_col.setSizePolicy(Policy.Expanding, Policy.Preferred)
        output_col.setSizePolicy(Policy.Expanding, Policy.Preferred)
        overlay_col.setSizePolicy(Policy.Expanding, Policy.Preferred)

        input_col.setMinimumWidth(self.PREVIEW + 40)
        output_col.setMinimumWidth(self.PREVIEW + 40)
        overlay_col.setMinimumWidth(self.PREVIEW + 40)

        controls_grid = QGridLayout()
        controls_grid.setContentsMargins(0, 0, 0, 0)
        controls_grid.setHorizontalSpacing(0)
        controls_grid.setVerticalSpacing(0)
        controls_grid.setColumnStretch(0, 1)
        controls_grid.setColumnStretch(1, 2)
        controls_grid.setColumnStretch(2, 1)

        left_tools.setContentsMargins(0, 0, 0, 0)

        left_tools_wrap = QWidget()
        left_tools_wrap.setLayout(left_tools)
        controls_grid.addWidget(left_tools_wrap, 0, 0, alignment=Qt.AlignmentFlag.AlignHCenter)

        controls_grid.addWidget(self.export_csv_btn, 0, 2,
                                alignment=Qt.AlignmentFlag.AlignHCenter)

        # === Create a horizontal layout to hold the title and Find ID row ===
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)

        header_row.addStretch()
        header_row.addWidget(self.table_title, alignment=Qt.AlignmentFlag.AlignVCenter)
        header_row.addWidget(left_tools_wrap, alignment=Qt.AlignmentFlag.AlignVCenter)
        header_row.addWidget(self.export_csv_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        header_row.addStretch()

        # --- Replace old controls_grid and title ---
        mid_box = QVBoxLayout()
        mid_box.setContentsMargins(0, 0, 0, 0)
        mid_box.setSpacing(2)
        mid_box.addLayout(header_row)
        mid_box.addWidget(self.table)

        mid_col = QWidget()
        mid_col.setLayout(mid_box)
        mid_col.setSizePolicy(Policy.Expanding, Policy.Preferred)
        mid_col.setMinimumWidth(self.PREVIEW + 40)

        self.table.setSizePolicy(Policy.Expanding, Policy.Expanding)

        # --- Top bar ---
        top_row = QHBoxLayout()
        top_row.addWidget(self.gpu_label)
        top_row.addSpacing(8)
        top_row.addWidget(self.gpu_mem_label)
        top_row.addStretch()
        top_row.addWidget(self.batch_size_label)
        top_row.addWidget(self.batch_size_spin)

        # --- Main layout with a 2x3 grid replacing the two HBox rows ---
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_row)
        main_layout.addSpacing(2)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(12)

        # Top row
        grid.addWidget(input_col,  0, 0)
        grid.addWidget(output_col, 0, 1)
        grid.addWidget(overlay_col, 0, 2)

        # Bottom row
        grid.addWidget(mid_col, 1, 0, 1, 3)

        grid.setColumnMinimumWidth(0, self.PREVIEW + 40)
        grid.setColumnMinimumWidth(1, self.PREVIEW + 40)
        grid.setColumnMinimumWidth(2, self.PREVIEW + 40)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)   # middle column wider for the table
        grid.setColumnStretch(2, 1)

        main_layout.addLayout(grid)
        main_layout.addSpacing(10)

        # --- Status bar ---
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_label.setStyleSheet("font-size: 10px; color: #495057; padding: 2px 6px; background: #f1f3f5;")
        self.status_label.setFixedHeight(26)

        self.cell_count_label = QLabel("🧫 Segmented Cells: 0")
        self.cell_count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.cell_count_label.setStyleSheet("font-size: 10px; color: #495057; padding: 2px 6px; background: #f1f3f5;")
        self.cell_count_label.setFixedHeight(26)

        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.cell_count_label)

        # main_layout.addSpacing(2)
        main_layout.addLayout(status_layout)

        self.setLayout(main_layout)

        # --- state ---
        self.input_image_path = None
        self.segmented_pixmap = None
        self.lab_ids = None
        self.cell_stats = None
        self.show_outline = False

        self.model = model
        self.method = method
        self.device = device
        self.parser_args = parser_args
        self.model_dict = model_dict
        self.params = params

        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(self.refresh_gpu_mem)
        self.gpu_timer.start(1000)
        self.refresh_gpu_mem()

        self.default_pixmap = QPixmap("cellembed sam.png").scaled(
            100, 100, Qt.AspectRatioMode.KeepAspectRatio
        )

        self.input_label.setPixmap(self.default_pixmap)
        self.output_label.setPixmap(self.default_pixmap)
        self.overlay_label.setPixmap(self.default_pixmap)

        self.setAcceptDrops(True)

        def _auto_min_width(btn, padding=32):
            fm = btn.fontMetrics()
            w = fm.horizontalAdvance(btn.text()) + padding
            btn.setMinimumWidth(w)
            btn.setMaximumWidth(16777215)

            Policy = getattr(QSizePolicy, "Policy", QSizePolicy)
            btn.setSizePolicy(Policy.Minimum, Policy.Fixed)

        for btn in [
            self.input_btn, self.segment_folder_btn, self.clear_btn,
            self.segment_btn, self.save_btn, self.outline_btn,
            self.save_overlay_btn, self.export_csv_btn
        ]:
            _auto_min_width(btn)

        from PyQt6.QtWidgets import QStyle

        self.input_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.segment_folder_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.clear_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self.segment_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.save_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.outline_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.save_overlay_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self.export_csv_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))

    def _fmt_bytes(self, n: int) -> str:
        return f"{n / (1024 ** 3):.2f} GB"

    def refresh_gpu_mem(self):
        try:
            if torch.cuda.is_available():
                idx = 0
                if hasattr(self, "device") and isinstance(self.device, torch.device) and self.device.type == "cuda":
                    idx = 0 if self.device.index is None else self.device.index

                free, total = torch.cuda.mem_get_info(idx)  # bytes
                allocated = torch.cuda.memory_allocated(idx)  # bytes
                reserved = torch.cuda.memory_reserved(idx)  # bytes

                used = total - free
                pct = (used / total) * 100 if total else 0.0

                text = (
                    f"VRAM: {self._fmt_bytes(used)} / {self._fmt_bytes(total)} "
                    f"({pct:.0f}%)  |  alloc {self._fmt_bytes(allocated)}, "
                    f"reserved {self._fmt_bytes(reserved)}"
                )
                self.gpu_mem_label.setText(text)
            else:
                self.gpu_mem_label.setText("VRAM: N/A")
        except Exception:
            self.gpu_mem_label.setText("VRAM: —")

    def load_input_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.tif)"
        )

        if file_path:
            self.input_image_path = file_path
            pixmap = QPixmap(file_path).scaled(
                self.PREVIEW, self.PREVIEW,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.input_label.setPixmap(pixmap)

            from PIL import Image
            with Image.open(file_path) as img:
                w, h = img.size
            self.input_res_label.setText(f"{w} × {h}")
            self.input_res_label.adjustSize()
            self.input_res_label.move(0, self.input_label.height() - self.input_res_label.height())

            self.output_label.setText("")
            self.overlay_label.setText("")
            self.status_label.setText(f"Image loaded ✔ ({w}×{h})")
            self.cell_count_label.setText("🧫 Segmented Cells: 0")

        if hasattr(self, "input_info_label"):
            self.input_info_label.hide()

        if hasattr(self, "output_info_label"):
            self.output_info_label.hide()

    def segment_folder(self):
        in_dir = QFileDialog.getExistingDirectory(self, "Select folder to segment", "")
        if not in_dir:
            return

        default_out = os.path.join(in_dir, "seg_results")
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder (or create new)", default_out)
        if not out_dir:
            out_dir = default_out
            os.makedirs(out_dir, exist_ok=True)

        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        file_paths = [
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not file_paths:
            self.status_label.setText("⚠ No images found in the selected folder.")
            return

        for w in [self.segment_btn, self.input_btn, self.segment_folder_btn, self.clear_btn]:
            w.setEnabled(False)
        self.status_label.setText(f"Batch started: {len(file_paths)} files")

        alpha = self.alpha_slider.value() / 100.0
        self.folder_thread = FolderSegThread(
            file_paths=file_paths,
            output_dir=out_dir,
            model=self.model,
            method=self.method,
            device=self.device,
            parser_args=self.parser_args,
            model_dict=self.model_dict,
            params=self.params,
            batch_size=int(self.batch_size_spin.value()),
            alpha=alpha
        )
        self.folder_thread.progress.connect(self._on_folder_progress)
        self.folder_thread.finished.connect(self._on_folder_finished)
        self.folder_thread.start()

    def _on_folder_progress(self, msg: str):
        self.status_label.setText(msg)

    def _on_folder_finished(self, msg: str):
        self.status_label.setText(msg)
        for w in [self.segment_btn, self.input_btn, self.segment_folder_btn, self.clear_btn]:
            w.setEnabled(True)

    def clear_all(self):
        self.input_label.setPixmap(self.default_pixmap)
        self.output_label.setPixmap(self.default_pixmap)
        self.overlay_label.setPixmap(self.default_pixmap)
        self.table.setRowCount(0)

        self.status_label.setText("Ready")
        self.cell_count_label.setText("🧫 Segmented Cells: 0")

        self.input_image_path = None
        self.segmented_pixmap = None
        self.lab_ids = None
        self.cell_stats = None
        self.base_lab_color = None
        self.lab_color = None
        self.base_img = None
        self.overlay_base = None
        self.show_outline = False
        self.input_res_label.clear()

        self._free_memory()

        if hasattr(self, "input_info_label"):
            self.input_info_label.show()

        if hasattr(self, "output_info_label"):
            self.output_info_label.show()

    def _center_input_info(self):
        parent = self.input_label
        label = self.input_info_label

        parent_w = parent.width()
        parent_h = parent.height()

        w = label.width()
        h = label.height()

        x = (parent_w - w) // 2
        y = 10

        label.move(x, y)

    def _center_output_info(self):

        parent = self.output_label
        label = self.output_info_label

        parent_w = parent.width()
        w = label.width()
        x = (parent_w - w) // 2
        y = 10

        label.move(x, y)


    def _free_memory(self):
        try:
            gc.collect()
            if torch.cuda.is_available():
                idx = 0
                if hasattr(self, "device") and isinstance(self.device, torch.device) and self.device.type == "cuda":
                    idx = 0 if self.device.index is None else self.device.index
                with torch.cuda.device(idx):
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    torch.cuda.synchronize()
            self.status_label.setText("Ready（memory cleared）")
        except Exception as e:
            self.status_label.setText(f"Ready（memory clear skipped: {e}）")

    def export_csv(self):
        if self.cell_stats is None:
            self.status_label.setText("⚠ No data to export!")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cell Stats", "", "CSV Files (*.csv)"
        )
        if save_path:
            self.cell_stats.to_csv(save_path, index=True)
            self.status_label.setText(f"CSV exported: {os.path.basename(save_path)} ✔")

    def run_segmentation(self):
        if not self.input_image_path:
            self.status_label.setText("⚠ No input image loaded!")
            return

        self.status_label.setText("Running segmentation ⏳ ...")
        QApplication.processEvents()

        self.thread = SegmentationThread(
            self.input_image_path,
            self.model,
            self.method,
            self.device,
            self.parser_args,
            self.model_dict,
            self.params,
            batch_size=int(self.batch_size_spin.value()),
        )
        self.thread.finished.connect(self.on_segmentation_finished)
        self.thread.start()

    def on_segmentation_finished(self, lab_color, lab_ids):
        self.lab_ids = lab_ids

        unique_ids = np.unique(self.lab_ids)
        unique_ids = unique_ids[unique_ids > 0]
        id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}
        new_lab = np.zeros_like(self.lab_ids, dtype=np.int32)
        for old_id, new_id in id_map.items():
            new_lab[self.lab_ids == old_id] = new_id
        self.lab_ids = new_lab

        from cellembed.utils.cellmorphology import CellImageAnalyzer
        analyzer = CellImageAnalyzer(image_array=self.lab_ids.astype(np.uint16))
        analyzer.compute_contours_and_stats()
        analyzer.calculate_morphology_stats()
        self.cell_stats = analyzer.generate_dataframe()

        colors = cc.cm.glasbey_bw_minc_20_minl_30_r.colors
        self.base_lab_color = label_to_color_image(self.lab_ids, colors=colors)

        self.show_output_image(self.base_lab_color)

        lab_color = np.ascontiguousarray(lab_color)
        h, w, c = lab_color.shape
        qimg = QImage(lab_color.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.PREVIEW, self.PREVIEW, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        self.segmented_pixmap = pixmap
        self.output_label.setPixmap(self.segmented_pixmap)

        from PIL import Image
        img = Image.open(self.input_image_path).convert("RGB")
        img = img.resize((w, h))
        img_np = np.array(img).astype(np.float32) / 255.0
        lab_color_f = lab_color.astype(np.float32) / 255.0

        alpha = 0.5
        mask = np.any(lab_color_f > 0, axis=-1, keepdims=True)
        overlay_np = img_np.copy()
        overlay_np[mask[..., 0]] = (1 - alpha) * img_np[mask[..., 0]] + alpha * lab_color_f[mask[..., 0]]

        overlay_np = (overlay_np * 255).astype(np.uint8)
        overlay_np = np.ascontiguousarray(overlay_np)

        qimg_overlay = QImage(overlay_np.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap_overlay = QPixmap.fromImage(qimg_overlay).scaled(
            self.PREVIEW, self.PREVIEW, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        self.overlay_label.setPixmap(pixmap_overlay)

        self.overlay_base = overlay_np.copy()
        self.current_alpha = 0.5

        self.base_img = np.array(img).astype(np.float32) / 255.0
        self.lab_color = lab_color.astype(np.float32) / 255.0

        self.update_overlay_alpha()
        self.update_table()

        unique_ids = np.unique(self.lab_ids)
        cell_count = len(unique_ids[unique_ids > 0])
        self.cell_count_label.setText(f"🧫 Segmented Cells: {cell_count}")

        self.status_label.setText("Segmentation completed ✔ (hover to check Cell ID)")

    def update_overlay_alpha(self):
        if getattr(self, "base_img", None) is None or getattr(self, "lab_color", None) is None:
            return

        alpha = self.alpha_slider.value() / 100.0
        self.current_alpha = alpha

        img_np = self.base_img
        lab_color_f = self.lab_color

        mask = np.any(lab_color_f > 0, axis=-1, keepdims=True)
        overlay_np = (1 - alpha) * img_np + alpha * lab_color_f
        overlay_np = np.where(mask, overlay_np, img_np)

        overlay_np = (overlay_np * 255).astype(np.uint8)
        h, w, _ = overlay_np.shape
        qimg_overlay = QImage(overlay_np.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap_overlay = QPixmap.fromImage(qimg_overlay).scaled(
            self.PREVIEW, self.PREVIEW, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        self.overlay_label.setPixmap(pixmap_overlay)

    def update_table(self):
        if self.cell_stats is None:
            return
        df = self.cell_stats
        self.table.setRowCount(len(df))
        for i, (idx, row) in enumerate(df.iterrows()):
            values = [
                str(idx),
                f"{row['cellArea']:.0f}",
                f"{row['cellCompactness']:.2f}",
                f"{row['cellCircularity']:.2f}",
                f"{row['cellSolidity']:.2f}",
                f"{row['cellEccentricity']:.2f}",
            ]
            for j, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, j, item)

    def save_output_image(self):
        if self.segmented_pixmap:
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Segmented Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)"
            )
            if save_path:
                self.segmented_pixmap.save(save_path)
                self.status_label.setText(f"Result saved: {os.path.basename(save_path)} ✔")
        else:
            self.status_label.setText("⚠ No segmented image to save!")

    def save_overlay_image(self):
        if getattr(self, "overlay_base", None) is None:
            self.status_label.setText("⚠ No overlay to save!")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Overlay Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        if save_path:
            alpha = self.alpha_slider.value() / 100.0
            img_np = self.base_img
            lab_color_f = self.lab_color
            mask = np.any(lab_color_f > 0, axis=-1, keepdims=True)

            overlay_np = (1 - alpha) * img_np + alpha * lab_color_f
            overlay_np = np.where(mask, overlay_np, img_np)
            overlay_np = (overlay_np * 255).astype(np.uint8)

            from PIL import Image
            Image.fromarray(overlay_np).save(save_path)

            self.status_label.setText(f"Overlay saved: {os.path.basename(save_path)} ✔")

    def update_display(self, lab_color):
        h, w, c = lab_color.shape
        if self.show_outline:
            outline_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            for cid in np.unique(self.lab_ids):
                if cid == 0:
                    continue
                mask = (self.lab_ids == cid).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(outline_img, contours, -1, (0, 0, 0), 1)
            lab_color = outline_img

        qimg = QImage(lab_color.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.PREVIEW, self.PREVIEW, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        self.segmented_pixmap = pixmap
        self.output_label.setPixmap(self.segmented_pixmap)

    def toggle_outline_mode(self):
        self.show_outline = not self.show_outline
        if self.lab_ids is not None:
            colors = cc.cm.glasbey_bw_minc_20_minl_30_r.colors
            lab_color = label_to_color_image(self.lab_ids, colors=colors)
            self.update_display(lab_color)

    def show_output_image(self, rgb_np):
        rgb_np = np.ascontiguousarray(rgb_np)
        h, w, _ = rgb_np.shape
        qimg = QImage(rgb_np.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.PREVIEW, self.PREVIEW, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.output_label.setPixmap(pixmap)

    def on_table_selection_changed(self):
        if self.lab_ids is None or self.base_lab_color is None:
            return
        items = self.table.selectedItems()
        if not items:
            self.show_output_image(self.base_lab_color)
            return

        row = items[0].row()
        cid_item = self.table.item(row, 0)
        if cid_item is None:
            return
        try:
            cid = int(cid_item.text())
        except ValueError:
            return

        vis = self.base_lab_color.copy()

        mask = (self.lab_ids == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

        self.show_output_image(vis)

    def eventFilter(self, source, event):
        if source == self.output_label and event.type() == event.Type.MouseMove:
            if self.lab_ids is None or self.output_label.pixmap() is None:
                return super().eventFilter(source, event)

            pos = event.position().toPoint()
            Lw, Lh = self.output_label.width(), self.output_label.height()
            pm = self.output_label.pixmap()
            Pw, Ph = pm.width(), pm.height()
            off_x = (Lw - Pw) // 2
            off_y = (Lh - Ph) // 2

            if not (off_x <= pos.x() < off_x + Pw and off_y <= pos.y() < off_y + Ph):
                self.status_label.setText("Ready")
                return True

            px = pos.x() - off_x
            py = pos.y() - off_y
            H, W = self.lab_ids.shape
            x = int(px * W / Pw)
            y = int(py * H / Ph)

            if 0 <= x < W and 0 <= y < H:
                cid = int(self.lab_ids[y, x])
                if cid == 0:
                    self.status_label.setText("Background")
                else:
                    if self.cell_stats is not None and cid in self.cell_stats.index:
                        stats = self.cell_stats.loc[cid]
                        area = stats["cellArea"]
                        com = stats["cellCompactness"]
                        circ = stats["cellCircularity"]
                        sol = stats["cellSolidity"]
                        ecc = stats["cellEccentricity"]
                        self.status_label.setText(
                            f"Cell {cid} | Area={area:.0f} | Com={com:.2f} | Circ={circ:.2f} | Sol={sol:.2f} | Ecc={ecc:.2f}"
                        )
                    else:
                        self.status_label.setText(f"Cell ID = {cid}")
            else:
                self.status_label.setText("Ready")
            return True

        return super().eventFilter(source, event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.splitext(file_path)[1].lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                self._load_input_path(file_path)
                break

    def _load_input_path(self, file_path):
        self.input_image_path = file_path
        pixmap = QPixmap(file_path).scaled(
            self.PREVIEW, self.PREVIEW, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.input_label.setPixmap(pixmap)

        from PIL import Image
        with Image.open(file_path) as img:
            w, h = img.size
        self.input_res_label.setText(f"{w} × {h}")
        self.input_res_label.adjustSize()
        self.input_res_label.move(0, self.input_label.height() - self.input_res_label.height())

        self.output_label.setPixmap(self.default_pixmap)
        self.overlay_label.setPixmap(self.default_pixmap)
        self.status_label.setText(f"Image loaded ✔ ({w}×{h})")
        self.cell_count_label.setText("🧫 Segmented Cells: 0")

        if hasattr(self, "input_info_label"):
            self.input_info_label.hide()

        if hasattr(self, "output_info_label"):
            self.output_info_label.hide()

    def find_cell_id(self):
        text = self.search_edit.text().strip()
        if not text:
            self.status_label.setText("Enter an integer ID")
            return
        try:
            cid = int(text)
        except ValueError:
            self.status_label.setText("ID must be an integer")
            return

        target_row = None
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.text() == str(cid):
                target_row = r
                break

        if target_row is None:
            self.status_label.setText(f"ID {cid} not found")
            return

        self.table.setCurrentCell(target_row, 0)
        self.table.selectRow(target_row)
        self.table.scrollToItem(self.table.item(target_row, 0))
        self.status_label.setText(f"Jumped to ID {cid}")
        self.on_table_selection_changed()

    def clear_search_highlight(self):
        if hasattr(self, "search_edit"):
            self.search_edit.clear()
        self.table.clearSelection()
        if getattr(self, "base_lab_color", None) is not None:
            self.show_output_image(self.base_lab_color)
        self.status_label.setText("Cleared")


def install_light_tech_style(app, gui):
    from PyQt6.QtWidgets import QStyleFactory
    app.setStyle(QStyleFactory.create("Fusion"))

    ACCENT1 = "#4F46E5"
    ACCENT2 = "#06B6D4"
    BG0     = "#F8FAFC"
    BG1     = "#FFFFFF"
    STROKE  = "rgba(15,23,42,0.08)"
    TEXT    = "#0F172A"
    MUTED   = "#64748B"

    try:
        fams = set(QFontDatabase.families())
        for f in ["Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Inter", "Segoe UI", "Helvetica", "Arial"]:
            if f in fams:
                app.setFont(QFont(f, 10))
                break
    except Exception:
        pass

    gui.setObjectName("MainWindow")
    for lab in [gui.input_title, gui.output_title, gui.overlay_title, gui.table_title]:
        lab.setObjectName("TitleLabel")
    for p in [gui.input_label, gui.output_label, gui.overlay_label]:
        p.setObjectName("ImagePanel")

    gui.table.setObjectName("CardTable")
    gui.gpu_label.setObjectName("Chip")
    gui.gpu_mem_label.setObjectName("Chip")
    gui.status_label.setObjectName("StatusPill")
    gui.cell_count_label.setObjectName("StatusPill")

    gui.input_btn.setProperty("variant", "primary")
    gui.segment_btn.setProperty("variant", "primary")
    gui.save_btn.setProperty("variant", "success")
    gui.save_overlay_btn.setProperty("variant", "success")
    gui.export_csv_btn.setProperty("variant", "secondary")
    gui.segment_folder_btn.setProperty("variant", "warning")
    gui.clear_btn.setProperty("variant", "danger")
    gui.outline_btn.setProperty("variant", "secondary")

    for w in [
        gui.input_btn, gui.segment_btn, gui.segment_folder_btn, gui.clear_btn,
        gui.outline_btn, gui.save_btn, gui.save_overlay_btn, gui.export_csv_btn,
        gui.input_label, gui.output_label, gui.overlay_label, gui.table,
        gui.gpu_label, gui.gpu_mem_label, gui.status_label, gui.cell_count_label,
        gui.batch_size_label, gui.alpha_label
    ]:
        w.setStyleSheet("")

    qss = f"""
    QWidget#MainWindow {{
        color: {TEXT};
        background-color: {BG0};
    }}

    QLabel#TitleLabel {{
        font-size: 15px; font-weight: 700; letter-spacing: 0.2px;
        color: {TEXT};
        padding: 2px 8px 4px 14px;
        border-left: 3px solid {ACCENT1};
    }}

    QLabel#Chip {{
        font-size: 12px; color: #0B1220;
        background-color: rgba(2,6,23,0.04);
        border: 1px solid {STROKE};
        padding: 4px 10px; border-radius: 8px;
    }}

    QLabel#StatusPill {{
        background-color: rgba(15,23,42,0.04);
        border: 1px solid {STROKE};
        color: {TEXT};
        padding: 6px 10px; border-radius: 10px;
    }}

    QLabel#ImagePanel {{
        background-color: {BG1};
        border: 1px solid {STROKE};
        border-radius: 14px;
    }}

    QPushButton {{
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: 600;
        border: 1px solid {STROKE};
        background-color: #ffffff;
        color: {TEXT};
    }}
    QPushButton:hover {{ border-color: rgba(15,23,42,0.18); }}

    QPushButton[variant="primary"] {{
        border: none; color: white;
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                         stop:0 {ACCENT1}, stop:1 {ACCENT2});
    }}
    QPushButton[variant="primary"]:hover {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                         stop:0 #4338CA, stop:1 #0891B2);
    }}

    QPushButton[variant="success"]       {{ background-color: #10B981; border: none; color: white; }}
    QPushButton[variant="success"]:hover {{ background-color: #059669; }}
    QPushButton[variant="warning"]       {{ background-color: #F59E0B; border: none; color: #111827; }}
    QPushButton[variant="warning"]:hover {{ background-color: #D97706; color: white; }}
    QPushButton[variant="danger"]        {{ background-color: #EF4444; border: none; color: white; }}
    QPushButton[variant="danger"]:hover  {{ background-color: #DC2626; }}
    QPushButton[variant="secondary"]     {{ background-color: #E5E7EB; color: #111827; }}
    QPushButton[variant="secondary"]:hover{{ background-color: #CBD5E1; }}

    QLineEdit {{
        background-color: #FFFFFF;
        border: 1px solid {STROKE};
        padding: 6px 10px; border-radius: 8px;
        color: {TEXT};
    }}
    QLineEdit:focus {{ border: 1px solid {ACCENT1}; }}

    QSpinBox {{
        background-color: #FFFFFF;
        border: 1px solid {STROKE}; border-radius: 8px; padding: 2px 8px;
        color: {TEXT};
    }}
    QSpinBox:focus {{ border: 1px solid {ACCENT1}; }}

    QSlider::groove:horizontal {{
        background-color: #E5E7EB; height: 6px; border-radius: 3px;
        border: 1px solid {STROKE};
    }}
    QSlider::sub-page:horizontal {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                         stop:0 {ACCENT1}, stop:1 {ACCENT2});
        border: none; height: 6px; border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background-color: white; width: 16px; height: 16px; margin: -6px 0;
        border-radius: 8px; border: 2px solid {ACCENT1};
    }}

    QTableWidget#CardTable {{
        background-color: {BG1};
        color: {TEXT};
        gridline-color: rgba(15,23,42,0.08);
        border: 1px solid {STROKE}; border-radius: 12px;
        alternate-background-color: #F8FAFC;
    }}
    QHeaderView::section {{
        background-color: #EEF2F7;
        color: #0F172A; font-weight: 700;
        border: 0px; border-bottom: 1px solid {STROKE};
        padding: 6px;
    }}
    QTableWidget::item {{
        selection-background-color: #E0EAFF;
        selection-color: #0F172A;
    }}

    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background-color: rgba(15,23,42,0.12);
        border-radius: 5px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background-color: rgba(15,23,42,0.22); }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    """
    app.setStyleSheet(qss)

    def add_shadow(w, blur=28, y=10, a=90):
        eff = QGraphicsDropShadowEffect()
        eff.setBlurRadius(blur)
        eff.setOffset(0, y)
        eff.setColor(QColor(2, 6, 23, a))
        w.setGraphicsEffect(eff)

    for w in [gui.input_label, gui.output_label, gui.overlay_label, gui.table]:
        add_shadow(w, blur=28, y=10, a=80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m_p", "--model_path", type=str, default=r"../cellembed/pretrained CEmbed-SAM")
    parser.add_argument("-m_f", "--model_folder", type=str, default="CEmbed SAM")
    parser.add_argument("-window", "--window_size", default=128, type=int)
    parser_args = parser.parse_args([])

    model, model_dict = load_model(path=parser_args.model_path, folder=parser_args.model_folder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    method = InstanSeg(
        binary_loss_fn_str=model_dict["binary_loss_fn"],
        seed_loss_fn=model_dict["seed_loss_fn"],
        n_sigma=model_dict["n_sigma"],
        cells_and_nuclei=model_dict["cells_and_nuclei"],
        to_centre=model_dict["to_centre"],
        window_size=parser_args.window_size,
        dim_coords=model_dict["dim_coords"],
        feature_engineering_function=model_dict["feature_engineering"],
    )
    method.initialize_pixel_classifier(model)
    model.eval().to(device)

    app = QApplication(sys.argv)
    gui = CellSegmentationGUI(model, method, device, parser_args, model_dict, params=None)
    install_light_tech_style(app, gui)
    gui.show()
    sys.exit(app.exec())
