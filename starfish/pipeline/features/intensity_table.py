from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from starfish.constants import Indices, AugmentedEnum, Coordinates


class IntensityTable(xr.DataArray):

    class Constants(AugmentedEnum):
        TILES = 'tiles'
        FEATURES = 'features'
        GENE = 'gene_name'  # TODO needs refactor to match gene_table constants
        QUALITY = 'quality'

    class SpotAttributes(AugmentedEnum):
        X = 'x'
        Y = 'y'
        Z = 'z'
        RADIUS = 'r'

    @classmethod
    def empty_intensity_table(cls, spot_attributes: pd.MultiIndex, n_ch: int, n_hyb: int) -> "IntensityTable":
        """Create an empty intensity table with pre-set axis whose values are zero

        Parameters
        ----------
        spot_attributes : pd.MultiIndex
        n_ch : int
        n_hyb : int

        Returns
        -------

        """
        cls._verify_spot_attributes(spot_attributes)
        channel_index = pd.Index(np.arange(n_ch))
        hyb_index = pd.Index(np.arange(n_hyb))
        data = np.zeros((spot_attributes.shape[0], n_ch, n_hyb))
        dims = (IntensityTable.Constants.FEATURES.value, Indices.CH.value, Indices.HYB.value)
        return cls(data=data, coords=(spot_attributes, channel_index, hyb_index), dims=dims)

    @staticmethod
    def _verify_spot_attributes(spot_attributes: pd.MultiIndex) -> None:
        """Run some checks on spot attributes"""
        if not isinstance(spot_attributes, pd.MultiIndex):
            raise ValueError(
                f'spot attributes must be a pandas MultiIndex, not {type(spot_attributes)}.')

        required_attributes = set(a.value for a in IntensityTable.SpotAttributes)
        missing_attributes = required_attributes.difference(spot_attributes.names)
        if missing_attributes:
            raise ValueError(
                f'Missing spot_attribute levels in provided MultiIndex: {missing_attributes}. '
                f'The following levels are required: {required_attributes}.')

    @classmethod
    def from_spot_data(
            cls, intensities: Union[xr.DataArray, np.ndarray], spot_attributes: pd.MultiIndex,
            *args, **kwargs) -> "IntensityTable":
        """Table to store image feature intensities and associated metadata

        Parameters
        ----------
        intensities : np.ndarray[Any]
            intensity data
        spot_attributes : pd.MultiIndex
            Name(s) of the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            are taken from ``coords`` (if possible) and otherwise default to
            ``['dim_0', ... 'dim_n']``.
        args :
            additional arguments to pass to the xarray constructor
        kwargs :
            additional keyword arguments to pass to the xarray constructor

        """

        if len(intensities.shape) != 3:
            raise ValueError(
                f'intensities must be a (features * ch * hyb) 3-d tensor. Provided intensities '
                f'shape ({intensities.shape}) is invalid.')

        cls._verify_spot_attributes(spot_attributes)

        coords = (
            (IntensityTable.Constants.FEATURES.value, spot_attributes),
            (Indices.CH.value, np.arange(intensities.shape[1])),
            (Indices.HYB.value, np.arange(intensities.shape[2]))
        )

        dims = (IntensityTable.Constants.FEATURES.value, Indices.CH.value, Indices.HYB.value)

        return cls(intensities, coords, dims, *args, **kwargs)

    def save(self, filename: str) -> None:
        """Save an IntensityTable as a Netcdf File

        Parameters
        ----------
        filename : str
            Name of Netcdf file

        """
        # TODO when https://github.com/pydata/xarray/issues/1077 (support for multiindex
        # serliazation) is merged, remove this reset_index() call and simplify load, below
        self.reset_index('features').to_netcdf(filename)

    @classmethod
    def load(cls, filename: str) -> "IntensityTable":
        """load an IntensityTable from Netcdf

        Parameters
        ----------
        filename : str
            File to load

        Returns
        -------
        IntensityTable

        """
        loaded = xr.open_dataarray(filename)
        intensity_table = cls(
            loaded.data,
            loaded.coords,
            loaded.dims
        )
        return intensity_table.set_index(features=list(intensity_table['features'].coords.keys()))

    def show(self, background_image: np.ndarray) -> None:
        """show spots on a background image"""
        raise NotImplementedError

    @classmethod
    def synthetic_intensities(
            cls, codebook, num_z: int=12, height: int=50, width: int=40, n_spots=10,
            mean_fluor_per_spot=200, mean_photons_per_fluor=50
    ) -> "IntensityTable":

        # TODO nsofroniew: right now there is no jitter on x-y positions of the spots
        z = np.random.randint(0, num_z, size=n_spots)
        y = np.random.randint(0, height, size=n_spots)
        x = np.random.randint(0, width, size=n_spots)
        r = np.empty(n_spots)

        names = [cls.SpotAttributes.Z.value, cls.SpotAttributes.Y.value,
                 cls.SpotAttributes.X.value, cls.SpotAttributes.RADIUS.value]
        spot_attributes = pd.MultiIndex.from_arrays([z, y, x, r], names=names)

        # empty data tensor
        data = np.zeros(shape=(n_spots, *codebook.shape[1:]))

        genes = np.random.choice(
            codebook.coords[cls.Constants.GENE.value], size=n_spots, replace=True)
        expected_bright_locations = np.where(codebook.loc[genes])

        # create a binary matrix where "on" spots are 1
        data[expected_bright_locations] = 1

        # add physical properties of fluorescence
        data *= np.random.poisson(mean_photons_per_fluor, size=data.shape)
        data *= np.random.poisson(mean_fluor_per_spot, size=data.shape)

        intensities = cls.from_spot_data(data, spot_attributes)
        intensities[cls.Constants.GENE.value] = ('features', genes)

        return intensities
