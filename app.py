from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import gui.gui_form
import image_manipulation.image_manipulation as im
import numpy as np


class ExampleApp(QtWidgets.QMainWindow, gui.gui_form.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.ImP = None
        self.image_height = None
        self.color_brightness_sliders = [self.redFactorSlider, self.greenFactorSlider, self.blueFactorSlider]
        self.color_brightness_line_edits = [self.redFactorLineEdit, self.greenFactorLineEdit, self.blueFactorLineEdit]

    # -------------------------------------------------------------------------
    # -------callback functions------------------------------------------------
    # -------------------------------------------------------------------------

    # -------global buttons----------------------------------------------------
    def on_load_image_button_clicked(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open File")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
        self.ImP = im.ImageProject.from_file(selected_files[0])
        self.image_height = self.ImP.image_original.shape[0]
        self.imageScalingSlider.setValue(100)
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

    # ------color saturation tab-----------------------------------------------
    def on_desaturation_factor_slider_changed(self):
        dsf = self.desaturationFactorSlider.value() / 100
        self.desaturationFactorLineEdit.setText(str(dsf))

    def on_desaturation_factor_slider_released(self):
        desaturation_factor = self.desaturationFactorSlider.value() / 100
        if self.ImP is not None:
            self.ImP.turn_all_towards_other(desaturation_factor)
            self.display_image()

    def on_desaturation_factor_input_edited(self):
        try:
            desaturation_factor = float(self.desaturationFactorLineEdit.text())
        except ValueError:
            desaturation_factor = 99.0
        if not 0.0 <= desaturation_factor <= 1.0:
            self.on_desaturation_factor_slider_changed()
        else:
            self.desaturationFactorSlider.setValue(int(desaturation_factor * 100))
            self.on_desaturation_factor_slider_released()

    # ------image scaling tab-----------------------------------------------
    def on_scaling_input_edited(self):
        try:
            scaling_factor = float(self.imageScalingLineEdit.text())
        except ValueError:
            scaling_factor = 99.0
        if not 0.0 <= scaling_factor <= 1.0:
            self.on_scaling_slider_changed()
        else:
            self.imageScalingSlider.setValue(int(scaling_factor * 100))
            self.on_scaling_slider_released()

    def on_scaling_slider_released(self):
        isf = self.imageScalingSlider.value() / 100
        if self.ImP is not None:
            self.image_height = int(self.ImP.image_original.shape[0] * isf)
            self.display_image()

    def on_scaling_slider_changed(self):
        isf = self.imageScalingSlider.value() / 100
        self.imageScalingLineEdit.setText(str(isf))

    def on_scale_to_window_button_clicked(self):
        if self.ImP is not None:
            self.image_height = int(self.graphicsView.height() * 0.99)
            isf = int(100 * self.image_height / self.ImP.image_original.shape[0])
            self.imageScalingSlider.setValue(isf)
            self.display_image()

    def on_original_size_button_clicked(self):
        if self.ImP is not None:
            self.imageScalingSlider.setValue(100)
            self.image_height = self.ImP.image_original.shape[0]
            self.display_image()


    # -------------------------------------------------------------------------
    # ---------utilities-------------------------------------------------------
    # -------------------------------------------------------------------------

    def display_image(self):
        img = (self.ImP.current_image * 255).astype(np.uint8)
        h, w, c = img.shape
        s = img.strides[0]
        qimg = QtGui.QImage(img, w, h, s, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaledToHeight(self.image_height)
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
