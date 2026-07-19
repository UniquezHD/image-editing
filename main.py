import os
import sys
import torch
os.environ["HF_HUB_OFFLINE"] = "1"
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

from PIL import Image

from diffusers import QwenImageEditPlusPipeline

from PySide6 import QtCore, QtWidgets, QtGui


class ImageLabel(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText('Drag Image here')
        self.setStyleSheet('''
            QLabel{
                           border: 4px dashed #aaa
                           }
                           ''')
    def update_pixmap(self, file_path, width=400, height=400):
        pixmap = QtGui.QPixmap(file_path)
        
        if pixmap.isNull():
            self.setText("Failed to load image")
            return 
        
        scaled_pixmap = pixmap.scaled(
            width, height, 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap) 



class Program(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.currentImagePath = ""
        self.prompt = ""

        self.runModelBtn = QtWidgets.QPushButton("Run")
        self.promptInput = QtWidgets.QLineEdit()
        self.promptInput.setMaximumHeight(30)
        self.dragDrop = ImageLabel()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.dragDrop)
        self.layout.addWidget(self.promptInput)
        self.layout.addWidget(self.runModelBtn)

        self.promptInput.textChanged.connect(self.HandlePromptOnChange)

        self.runModelBtn.clicked.connect(self.HandleRunModelBtn)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(QtCore.Qt.CopyAction)
            filePath = event.mimeData().urls()[0].toLocalFile()
            self.currentImagePath = filePath
            self.dragDrop.update_pixmap(filePath) 
            event.accept()
        else:
            event.ignore()

    @QtCore.Slot(str)
    def HandlePromptOnChange(self, text):
        self.prompt = text

    @QtCore.Slot()
    def HandleRunModelBtn(self):
        if self.currentImagePath:
            RunModelQwen(self.currentImagePath, self.prompt)
            #RunModelFlux(self.currentImagePath, self.prompt)
        else:
            print("Missing image")

def RunModelQwen(imagePath, prompt):
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "./Models/QwenImageEdit",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    print("pipeline loaded")

    pipeline.to('cuda')
    pipeline.set_progress_bar_config(disable=None)
    image = Image.open(imagePath)
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        output_image = output.images[0]
        output_image.save("output_image_edit_plus.png")
        print("image saved at", os.path.abspath("output_image_edit_plus.png"))


    print("done")

def RunModelFlux(imagePath, prompt):
    pipe = DiffusionPipeline.from_pretrained(
        "./Models/FLUX2",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    pipe.enable_model_cpu_offload()

    input_image = load_image(imagePath)

    image = pipe(
        image=input_image,
        prompt=prompt,
    ).images[0]

    image.save("output.png")
    print("Done")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = Program()
    widget.resize(600, 400)
    widget.show()

    sys.exit(app.exec())
