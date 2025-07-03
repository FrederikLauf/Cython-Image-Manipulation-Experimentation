from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import gui.gui_form
import image_manipulation.image_manipulation as im
import numpy as np


class ExampleApp(QtWidgets.QMainWindow, gui.gui_form.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.ImP = None
        self.color_brightness_sliders = [self.redFactorSlider, self.greenFactorSlider, self.blueFactorSlider]
        self.color_brightness_line_edits = [self.redFactorLineEdit, self.greenFactorLineEdit, self.blueFactorLineEdit]

    # -------------------------------------------------------------------------
    # -------callback functions------------------------------------------------
    # -------------------------------------------------------------------------

    # -------global buttons----------------------------------------------------
    def on_load_image_button_clicked(self):
        self.ImP = im.ImageProject.from_file(r"C:\Users\Frederik\Desktop\Raw\converted\IMG_2323.png")
        self.display_image()

    def on_show_histograms_button_clicked(self):
        self.ImP.histogram_plots()

    def on_show_scatter_plots_button_clicked(self):
        self.ImP.scatter_plots()
        self.ImP.scatter_plot_3d()

    # ------color brightness tab-----------------------------------------------
    def on_color_brightness_slider_changed(self):
        cf = [str(cbs.value() / 100.0) for cbs in self.color_brightness_sliders]
        _ = [cble.setText(val) for cble, val in zip(self.color_brightness_line_edits, cf)]

    def on_color_brightness_slider_released(self):
        color_factors = [cbs.value() / 100.0 for cbs in self.color_brightness_sliders]
        if self.ImP is not None:
            self.ImP.multiply_colors(color_factors)
            self.display_image()

    def on_color_brightness_input_edited(self):
        try:
            c_factors = [float(cble.text()) for cble in self.color_brightness_line_edits]
        except ValueError:
            c_factors = [-1.0]
        if not all(0.0 <= c <= 5.0 for c in c_factors):
            self.on_color_brightness_slider_changed()
        else:
            _ = [cbs.setValue(int(cf * 100)) for cbs, cf in zip(self.color_brightness_sliders, c_factors)]
            self.on_color_brightness_slider_released()

    # -------------------------------------------------------------------------
    # --------- actions--------------------------------------------------------
    # -------------------------------------------------------------------------

    def display_image(self):
        img = (self.ImP.image_working * 255).astype(np.uint8)
        h, w, c = img.shape
        s = img.strides[0]
        qimg = QtGui.QImage(img, w, h, s, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaledToHeight(450)
        scene = QtWidgets.QGraphicsScene(self)
        scene.addPixmap(pixmap)
        self.graphicsView.setScene(scene)


def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
