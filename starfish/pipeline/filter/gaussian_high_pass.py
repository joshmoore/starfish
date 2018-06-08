import argparse
from functools import partial
from numbers import Number
from typing import Callable, Union, Tuple

import numpy as np
from skimage import img_as_uint

from starfish.errors import DataFormatWarning
from starfish.io import Stack
from starfish.pipeline.filter.gaussian_low_pass import GaussianLowPass
from ._base import FilterAlgorithmBase


class GaussianHighPass(FilterAlgorithmBase):

    def __init__(
            self, sigma: Union[Number, Tuple[Number]], is_volume: bool=False, verbose: bool=False, **kwargs
    ) -> None:
        """Gaussian high pass filter

        Parameters
        ----------
        sigma : Union[Number, Tuple[Number]]
            standard deviation of gaussian kernel
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles independently.
        verbose : bool
            if True, report on filtering progress (default = False)

        """
        if isinstance(sigma, tuple):
            message = ("if passing an anisotropic kernel, the dimensionality must match the data shape ({shape}), not "
                       "{passed_shape}")
            if is_volume and len(sigma) != 3:
                raise ValueError(message.format(shape=3, passed_shape=len(sigma)))
            if not is_volume and len(sigma) != 2:
                raise ValueError(message.format(shape=2, passed_shape=len(sigma)))

        self.sigma = sigma
        self.is_volume = is_volume
        self.verbose = verbose

    @classmethod
    def get_algorithm_name(cls) -> str:
        return "gaussian_high_pass"

    @classmethod
    def add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--sigma", type=float, help="standard deviation of gaussian kernel")
        group_parser.add_argument(
            "--is-volume", action="store_true", help="indicates that the image stack should be filtered in 3d")

    @staticmethod
    def high_pass(image: np.ndarray, sigma: Union[Number, Tuple[Number]]) -> np.ndarray:
        """
        Applies a gaussian high pass filter to an image

        Parameters
        ----------
        image : numpy.ndarray[np.uint32]
            2-d or 3-d image data
        sigma : Union[Number, Tuple[Number]]
            Standard deviation of gaussian kernel

        Returns
        -------
        np.ndarray :
            Standard deviation of the Gaussian kernel that will be applied. If a float, an isotropic kernel will be
            assumed, otherwise the dimensions of the kernel give (z, y, x)

        """
        if image.dtype != np.uint16:
            DataFormatWarning('gaussian filters currently only support uint16 images. Image data will be converted.')
            image = img_as_uint(image)

        blurred: np.ndarray = GaussianLowPass.low_pass(image, sigma)

        over_flow_ind: np.ndarray[bool] = image < blurred
        filtered: np.ndarray = image - blurred
        filtered[over_flow_ind] = 0

        return filtered

    def filter(self, stack: Stack) -> None:
        """
        Perform in-place filtering of an image stack and all contained aux images.

        Parameters
        ----------
        stack : starfish.Stack
            Stack to be filtered.

        """

        high_pass: Callable = partial(self.high_pass, sigma=self.sigma)
        stack.image.apply(high_pass, is_volume=self.is_volume, verbose=self.verbose)

        # apply to aux dict too:
        for auxiliary_image in stack.auxiliary_images.values():
            auxiliary_image.apply(high_pass, is_volume=self.is_volume)
