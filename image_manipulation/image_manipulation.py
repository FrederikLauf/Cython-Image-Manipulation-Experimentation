import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import logging


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)
logger = logging.getLogger("ImageManipulation")

def log_info(text):
    logger.info(text)


class ImageProject:

    def __init__(self, image):
        log_info("init start")
        self.image_original = image.copy()
        self.image_saved = image.copy()
        self.current_image = image.copy()
        log_info("init stop")

    @classmethod
    def from_file(cls, path):
        log_info("loading image file")
        img = mpimg.imread(path)
        log_info("loaded image file")
        return cls(img)

    def _clip_upper(self):
        log_info("clipping to upper bound 1.0")
        self.current_image = np.clip(self.current_image, None, 1.0)
        log_info("clipped")

    def display(self):
        img_show = plt.imshow(self.current_image)
        plt.show()
        
    def display_original(self):
        img_show = plt.imshow(self.image_original)
        plt.show()

    def scatter_plots(self):
        log_info("configuring plot")
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
        log_info("separating linear spaces")
        c_lines = [self.current_image[:, :, i].flatten()[::500] for i in range(3)]
        log_info("plotting")
        axs[0].scatter(c_lines[0], c_lines[1], s=1, linewidths=0.1)
        axs[1].scatter(c_lines[1], c_lines[2], s=1, linewidths=0.1)
        axs[2].scatter(c_lines[0], c_lines[2], s=1, linewidths=0.1)
        log_info("showing plot")
        plt.show()

    def scatter_plot_3d(self):
        log_info("configuring plot")
        fig = plt.figure()
        axs = fig.add_subplot(projection="3d")
        axs.set_xlabel("red")
        axs.set_ylabel("green")
        axs.set_zlabel("blue")
        log_info("separating linear spaces")
        c_lines = [self.current_image[:, :, i].flatten()[::1000] for i in range(3)]
        log_info("plotting")
        axs.scatter(c_lines[0], c_lines[1], c_lines[2], s=1, linewidths=0.1)
        plt.show()

    def histogram_plots(self):
        log_info("configuring plot")
        fig, axs = plt.subplots(2, 2)
        br_ax, *c_axes = axs.flatten()
        br_ax.set_xlim((0, 1.01 * np.sqrt(3)))
        _ = [ax.set_xlim((0, 1.01)) for ax in c_axes]
        log_info("constructing brightness histogram")
        bm = np.linalg.norm(self.current_image, axis=2, keepdims=False)
        counts, bins = np.histogram(bm, bins=100, range=(0.0, np.sqrt(3)))
        log_info("plotting")
        br_ax.hist(bins[:-1], bins, weights=counts)
        log_info("constructing color histograms")
        c_lines = [self.current_image[:, :, i] for i in range(3)]
        c_histograms = [np.histogram(cl, bins=100, range=(0.0, 1.0)) for cl in c_lines]
        log_info("plotting")
        items = zip(c_axes, c_histograms, ('r', 'g', 'b'))
        _ = [ax.hist(ch[1][:-1], ch[1], weights=ch[0], color=col) for ax, ch, col in items]
        log_info("showing plot")
        plt.show()

    def multiply_colors(self, color_factors):
        log_info("multiplying colors")
        for i in range(3):
            self.current_image[:, :, i] = self.image_saved[:, :, i] * color_factors[i]
        log_info("multiplied colors")
        self._clip_upper()