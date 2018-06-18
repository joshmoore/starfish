import numpy as np
import pandas as pd
import pytest

from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector

from starfish.util.synthesize import _synthetic_spots
from starfish.constants import Indices
from starfish.munge import dataframe_to_multiindex
from starfish.pipeline.features.intensity_table import IntensityTable


@pytest.fixture(scope='function')
def small_intensity_table():
    intensities = np.array([
        [[0, 1],
         [1, 0]],
        [[1, 0],
         [0, 1]],
        [[0, 0],
         [1, 1]]
    ])

    spot_attributes = dataframe_to_multiindex(pd.DataFrame(
        data={
            IntensityTable.SpotAttributes.X: [0, 1, 2],
            IntensityTable.SpotAttributes.Y: [3, 4, 5],
            IntensityTable.SpotAttributes.Z: [0, 0, 0],
            IntensityTable.SpotAttributes.RADIUS: [0.1, 0.2, 0.3]
        }
    ))

    return IntensityTable.from_spot_data(intensities, spot_attributes)


# todo need more tests of this
def test_empty_intensity_table():
    x = [1, 2]
    y = [2, 3]
    z = [1, 1]
    r = [1, 1]
    spot_attributes = pd.MultiIndex.from_arrays([x, y, z, r], names=('x', 'y', 'z', 'r'))
    empty = IntensityTable.empty_intensity_table(spot_attributes, 2, 2)
    assert empty.shape == (2, 2, 2)


def test_fill_empty_intensity_table():
    pass


def test_intensity_table_raises_value_error_with_wrong_input_shape(small_intensity_table):
    pass


def test_intensity_table_raises_value_error_with_wrong_spot_attribute_type(small_intensity_table):
    pass


def test_intensity_table_raises_value_error_with_missing_spot_attributes(small_intensity_table):
    pass


def test_intensity_table_passes_args_and_kwargs_to_xarray_constructor(small_intensity_table):
    pass
