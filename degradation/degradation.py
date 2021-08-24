import cv2
import math
import numpy as np
import random
import torch
import imgaug.augmenters as ia
from PIL import Image

from scipy import special
from scipy import ndimage
from scipy.stats import multivariate_normal


class Degradation:
    def __init__(self):
        self.blur_kernel_size = 21
        self.kernel_list = [
            "iso",
            "aniso",
            "generalized_iso",
            "generalized_aniso",
            "plateau_iso",
            "plateau_aniso",
        ]
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 3]
        self.betap_range = [1, 2]
        self.sinc_prob = 0.1
        self.updown_type = ["up", "down", "keep"]
        self.mode_list = ["nearest", "bilinear", "bicubic"]
        self.resize_prob = [0.2, 0.7, 0.1]
        self.resize_range = [0.15, 1.5]

        # blur settings for the second degradation
        self.blur_kernel_size2 = 21
        self.kernel_list2 = [
            "iso",
            "aniso",
            "generalized_iso",
            "generalized_aniso",
            "plateau_iso",
            "plateau_aniso",
        ]
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1
        self.resize_prob2 = [0.3, 0.4, 0.3]
        self.resize_range2 = [0.3, 1.2]

        self.final_sinc_prob = 0.8

        self.kernel_range = [
            2 * v + 1 for v in range(3, 11)
        ]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def degradation_pipeline(self, image):
        """Generate kernels (used in the first degradation)"""
        self.ori_w, self.ori_h = image.size
        image = self.generate_kernel1(image)
        image = self.generate_sinc(image)
        image = self.random_resizing1(image)
        image = self.random_poisson_noise(image)
        image = self.random_gaussian_noise(image)
        image = self.random_jpeg_noise(image)

        """ Generate kernels (used in the second degradation) """
        image = self.generate_kernel2(image)
        image = self.random_resizing2(image)
        image = self.random_poisson_noise(image)
        image = self.random_gaussian_noise(image)
        if np.random.uniform() < 0.5:
            image = self.generate_sinc(image)
            image = self.random_jpeg_noise(image)
        else:
            image = self.random_jpeg_noise(image)
            image = self.generate_sinc(image)
        return Image.fromarray(image.clip(0, 255).astype(np.uint8))

    def random_poisson_noise(self, img, sigma_s="RAN", sigma_c="RAN"):
        min_log = np.log([0.0001])
        if sigma_s == "RAN":
            sigma_s = min_log + np.random.rand(1) * (np.log([0.16]) - min_log)
            sigma_s = np.exp(sigma_s)
            self.sigma_s = sigma_s
        if sigma_c == "RAN":
            sigma_c = min_log + np.random.rand(1) * (np.log([0.06]) - min_log)
            sigma_c = np.exp(sigma_c)
            self.sigma_c = sigma_c

        sigma_total = np.sqrt(sigma_s * img + sigma_c)
        """
        noisy_img = img + \
            np.random.normal(0.0, 1.0, img.shape) * (sigma_s * img) + \
            np.random.normal(0.0, 1.0, img.shape) * sigma_c
        """
        noisy_img = img + sigma_total * np.random.randn(img.shape[0], img.shape[1])
        noisy_img = np.clip(noisy_img, 0, 255)
        return noisy_img

    def random_gaussian_noise(self, image):
        noise_level = random.randint(1, 25)
        image = image + np.random.normal(0, noise_level, image.shape)
        return image.clip(min=0, max=255)

    def random_poisson_noise(self, image):
        sampled_lambda = len(np.unique(image))
<<<<<<< HEAD
        noise = (np.random.poisson(image) - image) * sampled_lambda / 100 # or 255?
=======
        noise = (np.random.poisson(image) - image) * sampled_lambda / 100  # or 255?
>>>>>>> c7728e6b4f18c4e365d9aae14f4cb8cfb41682d4
        image = image + noise
        return image.clip(min=0, max=255)

    def random_jpeg_noise(self, image):
        qf = random.randint(30, 95)
        trans = ia.JpegCompression(compression=qf)
        degrade_function = lambda x: trans.augment_image(x)
        image = degrade_function(image.astype(np.uint8))
        return image.clip(min=0, max=255)

    def random_resizing1(self, image):
        h, w, c = image.shape

        updown_type = random.choices(self.updown_type, self.resize_prob)
        mode = random.choice(self.mode_list)

        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1

        if mode == "nearest":
            flags = cv2.INTER_NEAREST
        elif mode == "bilinear":
            flags = cv2.INTER_LINEAR
        elif mode == "bicubic":
            flags = cv2.INTER_CUBIC

        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=flags)
        image = cv2.resize(image, (w, h), interpolation=flags)
        return image.clip(0, 255)

    def random_resizing2(self, image):
        h, w, c = image.shape
        updown_type = random.choices(self.updown_type, self.resize_prob2)
        mode = random.choice(self.mode_list)

        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1

        if mode == "nearest":
            flags = cv2.INTER_NEAREST
        elif mode == "bilinear":
            flags = cv2.INTER_LINEAR
        elif mode == "bicubic":
            flags = cv2.INTER_CUBIC

        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=flags)
        image = cv2.resize(image, (w, h), interpolation=flags)
        return image.clip(0, 255)

    def generate_kernel1(self, image):
        kernel_size = random.choice(self.kernel_range)
<<<<<<< HEAD
        # if np.random.uniform() < self.sinc_prob:
        #     # this sinc filter setting is for kernels ranging from [7, 21]
        #     if kernel_size < 13:
        #         omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     else:
        #         omega_c = np.random.uniform(np.pi / 5, np.pi)
        #     kernel = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        # else:
        kernel = self.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            self.betag_range,
            self.betap_range,
            noise_range=None)
=======
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = self.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
>>>>>>> c7728e6b4f18c4e365d9aae14f4cb8cfb41682d4

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        image = ndimage.filters.convolve(
            image, np.expand_dims(kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)

    def generate_kernel2(self, image):
        kernel_size = random.choice(self.kernel_range)
<<<<<<< HEAD
        # if np.random.uniform() < self.sinc_prob2:
        #     if kernel_size < 13:
        #         omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     else:
        #         omega_c = np.random.uniform(np.pi / 5, np.pi)
        #     kernel2 = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        # else:
        kernel2 = self.random_mixed_kernels(
            self.kernel_list2,
            self.kernel_prob2,
            kernel_size,
            self.blur_sigma2,
            self.blur_sigma2, [-math.pi, math.pi],
            self.betag_range2,
            self.betap_range2,
            noise_range=None)
=======
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = self.random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )
>>>>>>> c7728e6b4f18c4e365d9aae14f4cb8cfb41682d4

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        image = ndimage.filters.convolve(
            image, np.expand_dims(kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)

<<<<<<< HEAD
    def generate_sinc(self,image):
        # if np.random.uniform() < self.final_sinc_prob:
        #     kernel_size = random.choice(self.kernel_range)
        #     omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     sinc_kernel = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        #     sinc_kernel = torch.FloatTensor(sinc_kernel)
        # else:
        sinc_kernel = self.pulse_tensor
=======
    def generate_sinc(self, image):
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = self.circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
>>>>>>> c7728e6b4f18c4e365d9aae14f4cb8cfb41682d4

        image = ndimage.filters.convolve(
            image, np.expand_dims(sinc_kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)

    def sigma_matrix2(self, sig_x, sig_y, theta):
        """Calculate the rotated sigma matrix (two dimensional matrix).

        Args:
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.

        Returns:
            ndarray: Rotated sigma matrix.
        """
        D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
        U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.dot(U, np.dot(D, U.T))

    def mesh_grid(self, kernel_size):
        """Generate the mesh grid, centering at zero.

        Args:
            kernel_size (int):

        Returns:
            xy (ndarray): with the shape (kernel_size, kernel_size, 2)
            xx (ndarray): with the shape (kernel_size, kernel_size)
            yy (ndarray): with the shape (kernel_size, kernel_size)
        """
        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack(
            (
                xx.reshape((kernel_size * kernel_size, 1)),
                yy.reshape(kernel_size * kernel_size, 1),
            )
        ).reshape(kernel_size, kernel_size, 2)
        return xy, xx, yy

    def pdf2(self, sigma_matrix, grid):
        """Calculate PDF of the bivariate Gaussian distribution.

        Args:
            sigma_matrix (ndarray): with the shape (2, 2)
            grid (ndarray): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size.

        Returns:
            kernel (ndarrray): un-normalized kernel.
        """
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
        return kernel

    def cdf2(D, grid):
        """Calculate the CDF of the standard bivariate Gaussian distribution.
            Used in skewed Gaussian distribution.

        Args:
            D (ndarrasy): skew matrix.
            grid (ndarray): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size.

        Returns:
            cdf (ndarray): skewed cdf.
        """
        rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
        grid = np.dot(grid, D)
        cdf = rv.cdf(grid)
        return cdf

    def bivariate_Gaussian(
        self, kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True
    ):
        """Generate a bivariate isotropic or anisotropic Gaussian kernel.

        In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

        Args:
            kernel_size (int):
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
            grid (ndarray, optional): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size. Default: None
            isotropic (bool):

        Returns:
            kernel (ndarray): normalized kernel.
        """
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        kernel = self.pdf2(sigma_matrix, grid)
        kernel = kernel / np.sum(kernel)
        return kernel

    def bivariate_generalized_Gaussian(
        self, kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True
    ):
        """Generate a bivariate generalized Gaussian kernel.
            Described in `Parameter Estimation For Multivariate Generalized
            Gaussian Distributions`_
            by Pascal et. al (2013).

        In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

        Args:
            kernel_size (int):
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
            beta (float): shape parameter, beta = 1 is the normal distribution.
            grid (ndarray, optional): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size. Default: None

        Returns:
            kernel (ndarray): normalized kernel.

        .. _Parameter Estimation For Multivariate Generalized Gaussian
        Distributions: https://arxiv.org/abs/1302.6498
        """
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(
            -0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta)
        )
        kernel = kernel / np.sum(kernel)
        return kernel

    def bivariate_plateau(
        self, kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True
    ):
        """Generate a plateau-like anisotropic kernel.
        1 / (1+x^(beta))

        Ref: https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution

        In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

        Args:
            kernel_size (int):
            sig_x (float):
            sig_y (float):
            theta (float): Radian measurement.
            beta (float): shape parameter, beta = 1 is the normal distribution.
            grid (ndarray, optional): generated by :func:`mesh_grid`,
                with the shape (K, K, 2), K is the kernel size. Default: None

        Returns:
            kernel (ndarray): normalized kernel.
        """
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.reciprocal(
            np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1
        )
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_bivariate_Gaussian(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        noise_range=None,
        isotropic=True,
    ):
        """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.

        In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

        Args:
            kernel_size (int):
            sigma_x_range (tuple): [0.6, 5]
            sigma_y_range (tuple): [0.6, 5]
            rotation range (tuple): [-math.pi, math.pi]
            noise_range(tuple, optional): multiplicative kernel noise,
                [0.75, 1.25]. Default: None

        Returns:
            kernel (ndarray):
        """
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        kernel = self.bivariate_Gaussian(
            kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic
        )

        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_bivariate_generalized_Gaussian(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        beta_range,
        noise_range=None,
        isotropic=True,
    ):
        """Randomly generate bivariate generalized Gaussian kernels.

        In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

        Args:
            kernel_size (int):
            sigma_x_range (tuple): [0.6, 5]
            sigma_y_range (tuple): [0.6, 5]
            rotation range (tuple): [-math.pi, math.pi]
            beta_range (tuple): [0.5, 8]
            noise_range(tuple, optional): multiplicative kernel noise,
                [0.75, 1.25]. Default: None

        Returns:
            kernel (ndarray):
        """
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        # assume beta_range[0] < 1 < beta_range[1]
        if np.random.uniform() < 0.5:
            beta = np.random.uniform(beta_range[0], 1)
        else:
            beta = np.random.uniform(1, beta_range[1])

        kernel = self.bivariate_generalized_Gaussian(
            kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic
        )

        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_bivariate_plateau(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        beta_range,
        noise_range=None,
        isotropic=True,
    ):
        """Randomly generate bivariate plateau kernels.

        In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

        Args:
            kernel_size (int):
            sigma_x_range (tuple): [0.6, 5]
            sigma_y_range (tuple): [0.6, 5]
            rotation range (tuple): [-math.pi/2, math.pi/2]
            beta_range (tuple): [1, 4]
            noise_range(tuple, optional): multiplicative kernel noise,
                [0.75, 1.25]. Default: None

        Returns:
            kernel (ndarray):
        """
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        # TODO: this may be not proper
        if np.random.uniform() < 0.5:
            beta = np.random.uniform(beta_range[0], 1)
        else:
            beta = np.random.uniform(1, beta_range[1])

        kernel = self.bivariate_plateau(
            kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic
        )
        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)

        return kernel

    def random_mixed_kernels(
        self,
        kernel_list,
        kernel_prob,
        kernel_size=21,
        sigma_x_range=[0.6, 5],
        sigma_y_range=[0.6, 5],
        rotation_range=[-math.pi, math.pi],
        betag_range=[0.5, 8],
        betap_range=[0.5, 8],
        noise_range=None,
    ):
        """Randomly generate mixed kernels.

        Args:
            kernel_list (tuple): a list name of kenrel types,
                support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso',
                'plateau_aniso']
            kernel_prob (tuple): corresponding kernel probability for each
                kernel type
            kernel_size (int):
            sigma_x_range (tuple): [0.6, 5]
            sigma_y_range (tuple): [0.6, 5]
            rotation range (tuple): [-math.pi, math.pi]
            beta_range (tuple): [0.5, 8]
            noise_range(tuple, optional): multiplicative kernel noise,
                [0.75, 1.25]. Default: None

        Returns:
            kernel (ndarray):
        """
        kernel_type = random.choices(kernel_list, kernel_prob)[0]
        if kernel_type == "iso":
            kernel = self.random_bivariate_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                noise_range=noise_range,
                isotropic=True,
            )
        elif kernel_type == "aniso":
            kernel = self.random_bivariate_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                noise_range=noise_range,
                isotropic=False,
            )
        elif kernel_type == "generalized_iso":
            kernel = self.random_bivariate_generalized_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betag_range,
                noise_range=noise_range,
                isotropic=True,
            )
        elif kernel_type == "generalized_aniso":
            kernel = self.random_bivariate_generalized_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betag_range,
                noise_range=noise_range,
                isotropic=False,
            )
        elif kernel_type == "plateau_iso":
            kernel = self.random_bivariate_plateau(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betap_range,
                noise_range=None,
                isotropic=True,
            )
        elif kernel_type == "plateau_aniso":
            kernel = self.random_bivariate_plateau(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betap_range,
                noise_range=None,
                isotropic=False,
            )
        return kernel

    # np.seterr(divide='ignore', invalid='ignore')

    def circular_lowpass_kernel(self, cutoff, kernel_size, pad_to=0):
        """2D sinc filter, ref: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

        Args:
            cutoff (float): cutoff frequency in radians (pi is max)
            kernel_size (int): horizontal and vertical size, must be odd.
            pad_to (int): pad kernel size to desired size, must be odd or zero.
        """
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        kernel = np.fromfunction(
            lambda x, y: cutoff
            * special.j1(
                cutoff
                * np.sqrt(
                    (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2
                )
            )
            / (
                2
                * np.pi
                * np.sqrt(
                    (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2
                )
            ),
            [kernel_size, kernel_size],
        )
        kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff ** 2 / (
            4 * np.pi
        )
        kernel = kernel / np.sum(kernel)
        if pad_to > kernel_size:
            pad_size = (pad_to - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return kernel
