from itertools import product
import numpy as np
import pandas as pd
from skimage.feature import blob_log

from starfish.constants import Indices
from starfish.pipeline.features.intensity_table import IntensityTable
from starfish.image import ImageStack
from starfish.util.argparse import FsExistsType
from starfish.munge import dataframe_to_multiindex
from ._base import SpotFinderAlgorithmBase


class GaussianSpotDetector(SpotFinderAlgorithmBase):

    def __init__(
            self, min_sigma, max_sigma, num_sigma, threshold,
            blobs_stack, overlap=0.5, measurement_type='max', is_volume: bool=True, **kwargs):
        """Multi-dimensional gaussian spot detector

        Parameters
        ----------
        min_sigma : float
            The minimum standard deviation for Gaussian Kernel. Keep this low to
            detect smaller blobs.
        max_sigma : float
            The maximum standard deviation for Gaussian Kernel. Keep this high to
            detect larger blobs.
        num_sigma : int
            The number of intermediate values of standard deviations to consider
            between `min_sigma` and `max_sigma`.
        threshold : float
            The absolute lower bound for scale space maxima. Local maxima smaller
            than thresh are ignored. Reduce this to detect blobs with less
            intensities.
        overlap : float [0, 1]
            If two spots have more than this fraction of overlap, the spots are combined (default = 0.5)
        blobs_stack : Union[ImageStack, str]
            ImageStack or the path or URL that references the ImageStack that contains the blobs.
        measurement_type : str ['max', 'mean']
            name of the function used to calculate the intensity for each identified spot area

        Notes
        -----
        This spot detector is very sensitive to the threshold that is selected, and the threshold is defined as an
        absolute value -- therefore it must be adjusted depending on the datatype of the passed image.


        """
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.is_volume = is_volume
        if isinstance(blobs_stack, ImageStack):
            self.blobs_stack = blobs_stack
        else:
            self.blobs_stack = ImageStack.from_path_or_url(blobs_stack)

        try:
            self.measurement_function = getattr(np, measurement_type)
        except AttributeError:
            raise ValueError(
                f'measurement_type must be a numpy reduce function such as "max" or "mean". {measurement_type} '
                f'not found.')

    @staticmethod
    def _measure_blob_intensity(image, spots, measurement_function) -> pd.Series:
        # todo these values are messed up
        def fn(row):
            x_min = int(round(row.x_min))
            x_max = int(round(row.x_max))
            y_min = int(round(row.y_min))
            y_max = int(round(row.y_max))
            z_min = int(round(row.z_min))
            z_max = int(round(row.z_max))
            return measurement_function(image[z_min:z_max, y_min:y_max, x_min:x_max])
        return spots.apply(
            fn,
            axis=1
        )

    def _measure_spot_intensities(self, stack, spot_attributes):
        tile_iterator = (
            stack.get_slice({Indices.CH: c, Indices.HYB: h}) for h, c in
            product(range(stack.shape[Indices.CH]),
                    range(stack.shape[Indices.HYB])))
        intensities = {
            tile_data: self._measure_blob_intensity(
                image, spot_attributes, self.measurement_function)
            for image, tile_data in tile_iterator
        }

        n_ch = stack.shape[Indices.CH]
        n_hyb = stack.shape[Indices.HYB]

        spot_attribute_index = dataframe_to_multiindex(spot_attributes)
        intensity_table = IntensityTable.empty_intensity_table(spot_attribute_index, n_ch, n_hyb)

        for (ch, hyb), values in intensities.items():
            # ch, hyb = tile_data.iloc[i, :]
            intensity_table.loc[:, ch, hyb] = values

        return intensity_table

    def _find_spot_locations(self, blobs_image: np.ndarray) -> pd.DataFrame:

        # find blobs
        fitted_blobs_array: np.ndarray = blob_log(
            blobs_image, self.min_sigma, self.max_sigma, self.num_sigma, self.threshold,
            self.overlap)
        # convert to DataFrame
        fitted_blobs = pd.DataFrame(data=fitted_blobs_array, columns=['z', 'y', 'x', 'r'])

        if fitted_blobs.shape[0] == 0:
            raise ValueError('No spots detected with provided parameters')

        # convert standard deviation of gaussian kernel used to identify spot to radius of spot
        fitted_blobs['r'] *= np.sqrt(3)

        fitted_blobs[['z', 'y', 'x']] = fitted_blobs[['z', 'y', 'x']].astype(int)

        fitted_blobs['x_min'] = np.clip(np.floor(fitted_blobs.x - fitted_blobs.r), 0, None)
        fitted_blobs['x_max'] = np.clip(np.ceil(fitted_blobs.x + fitted_blobs.r),
                                        None, blobs_image.shape[2])
        fitted_blobs['y_min'] = np.clip(np.floor(fitted_blobs.y - fitted_blobs.r), 0, None)
        fitted_blobs['y_max'] = np.clip(np.ceil(fitted_blobs.y + fitted_blobs.r),
                                        None, blobs_image.shape[1])
        fitted_blobs['z_min'] = np.clip(np.floor(fitted_blobs.z - fitted_blobs.r), 0, None)
        fitted_blobs['z_max'] = np.clip(np.ceil(fitted_blobs.z + fitted_blobs.r),
                                        None, blobs_image.shape[0])

        fitted_blobs['intensity'] = self._measure_blob_intensity(
            blobs_image, fitted_blobs, self.measurement_function)
        fitted_blobs['spot_id'] = np.arange(fitted_blobs.shape[0])

        return fitted_blobs

    def find(self, hybridization_image) -> IntensityTable:
        blobs = self.blobs_stack.max_proj(Indices.HYB, Indices.CH)
        spot_attributes = self._find_spot_locations(blobs)
        intensity_table = self._measure_spot_intensities(hybridization_image, spot_attributes)
        return intensity_table

    @classmethod
    def get_algorithm_name(cls):
        return 'GaussianSpotDetector'

    @classmethod
    def add_arguments(cls, group_parser):
        group_parser.add_argument("--blobs-stack", type=FsExistsType(), required=True)
        group_parser.add_argument(
            "--min-sigma", default=4, type=int, help="Minimum spot size (in standard deviation)")
        group_parser.add_argument(
            "--max-sigma", default=6, type=int, help="Maximum spot size (in standard deviation)")
        group_parser.add_argument("--num-sigma", default=20, type=int, help="Number of sigmas to try")
        group_parser.add_argument("--threshold", default=.01, type=float, help="Dots threshold")
        group_parser.add_argument(
            "--overlap", default=0.5, type=float, help="dots with overlap of greater than this fraction are combined")
        group_parser.add_argument(
            "--show", default=False, action='store_true', help="display results visually")
