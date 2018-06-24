from typing import Tuple

from starfish.pipeline.features.codebook import Codebook
from starfish.pipeline.features.intensity_table import IntensityTable
from starfish.image import ImageStack


class SyntheticData:

    def __init__(
            self,
            n_hyb: int=4,
            n_ch: int=4,
            n_z: int=10,
            height: int=50,
            width: int=45,
            n_codes: int=16,
            n_spots: int=20,
            n_photons_background: int=1000,
            background_electrons: int=1,
            point_spread_function: Tuple[int, ...]=(4, 2, 2),
            camera_detection_efficiency: float=0.25,
            gray_level: float=37000.0 / 2 ** 16,
            ad_conversion_bits: int=16,
            mean_fluor_per_spot: int=200,
            mean_photons_per_fluor: int=50,
            fill_dynamic_range: bool=True,

    ) -> None:
        self.n_hyb = n_hyb
        self.n_ch = n_ch
        self.n_z = n_z
        self.height = height
        self.width = width
        self.n_codes = n_codes
        self.n_spots = n_spots
        self.n_photons_background = n_photons_background
        self.background_electrons = background_electrons
        self.point_spread_function = point_spread_function
        self.camera_detection_efficiency = camera_detection_efficiency
        self.gray_level = gray_level
        self.ad_coversion_bits = ad_conversion_bits
        self.mean_fluor_per_spot = mean_fluor_per_spot
        self.mean_photons_per_fluor = mean_photons_per_fluor
        self.fill_dynamic_range = fill_dynamic_range

    def codebook(self) -> Codebook:
        return Codebook.synthetic_one_hot_codebook(self.n_hyb, self.n_ch, self.n_codes)

    def intensities(self, codebook=None) -> IntensityTable:
        if codebook is None:
            codebook = self.codebook()
        return IntensityTable.synthetic_intensities(
            codebook, self.n_z, self.height, self.width, self.n_spots,
            self.mean_fluor_per_spot, self.mean_photons_per_fluor)

    def spots(self, intensities=None) -> ImageStack:
        if intensities is None:
            intensities = self.intensities()
        return ImageStack.synthetic_spots(
            intensities, self.n_z, self.height, self.width, self.n_photons_background,
            self.point_spread_function, self.camera_detection_efficiency,
            self.background_electrons, self.gray_level, self.ad_coversion_bits,
            fill_dynamic_range=self.fill_dynamic_range)
