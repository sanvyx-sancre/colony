import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit,
    QHBoxLayout, QLineEdit, QPushButton, QCheckBox, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QWheelEvent, QIcon
import cv2

# add src/ to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import run_inference, OUTPUT_DIR, model, CONF_THRESHOLD, MERGE_IOU_THRESHOLD


class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def display_image(self, path):
        pixmap = QPixmap(str(path))
        if not self.pixmap_item:
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene().addItem(self.pixmap_item)
        else:
            self.pixmap_item.setPixmap(pixmap)

        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        if not self.pixmap_item:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


class BacteriaCounterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bakteri Koloni Sayacı")
        self.resize(900, 700)
        self.setAcceptDrops(True)

        self.current_image_path = None
        self.gallery_paths = []

        main_layout = QVBoxLayout(self)

        # info label
        self.info_label = QLabel("Buraya görüntüyü sürükleyip bırakın")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.info_label)

        # controls
        control_layout = QHBoxLayout()

        self.conf_input = QLineEdit(str(CONF_THRESHOLD))
        self.merge_iou_input = QLineEdit(str(MERGE_IOU_THRESHOLD))
        self.dark_checkbox = QCheckBox("Karanlık mod")
        self.dark_checkbox.stateChanged.connect(self.toggle_dark_mode)

        control_layout.addWidget(QLabel("Güven:"))
        control_layout.addWidget(self.conf_input)
        control_layout.addWidget(QLabel("Merge IoU:")) # Intersection Over Union
        control_layout.addWidget(self.merge_iou_input)
        control_layout.addWidget(self.dark_checkbox)

        main_layout.addLayout(control_layout)

        # results
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)

        # image viewer
        self.image_viewer = ImageViewer()
        main_layout.addWidget(self.image_viewer, stretch=3)

        # save button
        self.save_button = QPushButton("Mevcut Görüntüyü Kaydet...")
        self.save_button.clicked.connect(self.save_image)
        main_layout.addWidget(self.save_button)

        # gallery
        self.gallery_list = QListWidget()
        self.gallery_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery_list.setIconSize(QPixmap(100, 100).size())
        self.gallery_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery_list.itemClicked.connect(self.gallery_item_clicked)
        main_layout.addWidget(self.gallery_list, stretch=1)

    # ---------------- logic ----------------

    def toggle_dark_mode(self, state):
        if state == Qt.CheckState.Checked.value:
            self.setStyleSheet("""
                QWidget { background-color: #2b2b2b; color: #f0f0f0; }
                QLineEdit, QTextEdit { background-color: #3c3c3c; color: #f0f0f0; }
            """)
        else:
            self.setStyleSheet("")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        self.result_text.clear()

        try:
            conf = float(self.conf_input.text())
            merge_iou = float(self.merge_iou_input.text())
        except ValueError:
            self.result_text.append("[!] invalid threshold input")
            return

        # 🔥 THIS IS THE IMPORTANT PART
        model.model.conf = conf
        model.model.iou = merge_iou

        for url in urls:
            file_path = Path(url.toLocalFile())
            self.info_label.setText(f"Processing: {file_path.name}")

            try:
                run_inference(
                    file_path,
                    conf=conf,
                    iou_thresh=0.4,      # model nms iou
                    merge_iou=merge_iou  # your custom merge
                )

                self.result_text.append(f"[✓] Processed: {file_path.name}")

                processed_path = OUTPUT_DIR / file_path.name
                if processed_path.exists():
                    self.gallery_paths.append(processed_path)
                    self.add_to_gallery(processed_path)
                    self.display_image(processed_path)
                    self.current_image_path = processed_path
                else:
                    self.result_text.append("[!] processed image not found")

            except Exception as e:
                self.result_text.append(f"[!] ERROR: {e}")

        self.info_label.setText("Buraya görüntüyü sürükleyip bırakın")

    def display_image(self, path):
        self.image_viewer.display_image(path)

    def add_to_gallery(self, path):
        item = QListWidgetItem()
        pix = QPixmap(str(path)).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
        item.setIcon(QIcon(pix))
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.gallery_list.addItem(item)

    def gallery_item_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.current_image_path = path
        self.display_image(path)

    def save_image(self):
        if not self.current_image_path:
            self.result_text.append("[!] no image to save")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Görüntüyü Kaydet", str(self.current_image_path),
            "Images (*.png *.jpg *.jpeg)"
        )

        if save_path:
            img = cv2.imread(str(self.current_image_path))
            cv2.imwrite(save_path, img)
            self.result_text.append(f"[✓] kaydedildi: {save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BacteriaCounterGUI()
    window.show()
    sys.exit(app.exec())
