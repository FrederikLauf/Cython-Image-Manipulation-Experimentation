import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from image_manipulation.cynalg import cylantro
import numpy as np
import logging


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)
logger = logging.getLogger("ImageManipulation")

def log_info(text):
    logger.info(text)


class ImageProject:

    def __init__(self, image):
        self.image_original = image.copy()
        self.image_saved = image.copy()
        self.current_image = image.copy()

    @classmethod
    def from_file(cls, path):
        pil_image = Image.open(path)
        img = (np.asarray(pil_image) / 255).astype(np.double)
        M, N = img.shape[0], img.shape[1]
        pil_image_red = pil_image.resize((750, int(M* 750 / N)), resample=Image.Resampling.LANCZOS)
        img_reduced = (np.asarray(pil_image_red) / 255).astype(np.double)
        return cls(img)

    def _clip(self):
        self.current_image = np.clip(self.current_image, 0.0, 1.0)

    def display(self):
        img_show = plt.imshow(self.current_image)
        plt.show()
        
    def display_original(self):
        img_show = plt.imshow(self.image_original)
        plt.show()

    def scatter_plots(self):
        fig, axs = plt.subplots(1, 3)
        _ = [ax.set_xlim((0, 1.01)) for ax in axs]
        _ = [ax.set_ylim((0, 1.01)) for ax in axs]
        _ = [ax.set_aspect("equal") for ax in axs]
        axs[0].set_xlabel("red")
        axs[0].set_ylabel("green")
        axs[1].set_xlabel("green")
        axs[1].set_ylabel("blue")
        axs[2].set_xlabel("red")
        axs[2].set_ylabel("blue")
        c_lines = [self.current_image[:, :, i].flatten()[::500] for i in range(3)]
        axs[0].scatter(c_lines[0], c_lines[1], s=1, linewidths=0.1)
        axs[1].scatter(c_lines[1], c_lines[2], s=1, linewidths=0.1)
        axs[2].scatter(c_lines[0], c_lines[2], s=1, linewidths=0.1)
        plt.show()

    def scatter_plot_3d(self):
        fig = plt.figure()
        axs = fig.add_subplot(projection="3d")
        axs.set_xlabel("red")
        axs.set_ylabel("green")
        axs.set_zlabel("blue")
        c_lines = [self.current_image[:, :, i].flatten()[::1000] for i in range(3)]
        axs.scatter(c_lines[0], c_lines[1], c_lines[2], s=1, linewidths=0.1)
        plt.show()

    def histogram_plots(self):
        fig, axs = plt.subplots(2, 2)
        br_ax, *c_axes = axs.flatten()
        br_ax.set_xlim((0, 1.01 * np.sqrt(3)))
        _ = [ax.set_xlim((0, 1.01)) for ax in c_axes]
        bm = np.linalg.norm(self.current_image, axis=2, keepdims=False)
        counts, bins = np.histogram(bm, bins=100, range=(0.0, np.sqrt(3)))
        br_ax.hist(bins[:-1], bins, weights=counts)
        c_lines = [self.current_image[:, :, i] for i in range(3)]
        c_histograms = [np.histogram(cl, bins=100, range=(0.0, 1.0)) for cl in c_lines]
        items = zip(c_axes, c_histograms, ('r', 'g', 'b'))
        _ = [ax.hist(ch[1][:-1], ch[1], weights=ch[0], color=col) for ax, ch, col in items]
        plt.show()

    def multiply_colors(self, color_factors):
        cylantro.scale_all_in_components(self.image_original, self.current_image, color_factors[0], color_factors[1], color_factors[2])

    def turn_all_towards_other(self, factor, other):
        cylantro.turn_all_towards_other(self.image_original, self.current_image, other, factor)
        
    def turn_all_awayfrom_other(self, factor, other):
        cylantro.turn_all_awayfrom_other(self.image_original, self.current_image, other, factor)
        self._clip()

    def rotate_all_constant(self, axis, angle):
        axis = np.array([0, 0, 1], dtype=self.image_original.dtype)
        angle = 0.2
        cylantro.rotate_all_constant(self.image_original, self.current_image, axis, angle)