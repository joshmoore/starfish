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
    r = [1, 1]
    spot_attributes = pd.MultiIndex.from_arrays([x, y, r], names=('x', 'y', 'r'))
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



# @pytest.mark.skip('needs codebook and data generated synthetically')
# def test_intensity_table():
#     data, codebook = _synthetic_spots()
#     gsd = GaussianSpotDetector(blobs_image_name='dots', min_sigma=2, max_sigma=10, num_sigma=10, threshold=0.1)
#     intensity_table = gsd.find(data)
#     codebook = Codebook.from_code_array(codebook, 4, 4)
#     result = codebook.decode_euclidean(intensity_table)
#
#     expected_gene_decoding = np.array([77, 87, 27, 77, 74, 94, 106, 77, 32, 110, 21, 46, 103, 39, 85, 108, 11, 4])
#     assert np.array_equal(result.indexes['features'].get_level_values('gene'), expected_gene_decoding)
#
#

