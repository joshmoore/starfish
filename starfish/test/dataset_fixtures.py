from copy import deepcopy

import pytest

from starfish.io import Stack
from starfish.constants import Indices
from starfish.image import ImageStack
from starfish.util import synthesize


# TODO ambrosejcarr: all fixtures should emit a stack and a codebook
@pytest.fixture(scope='session')
def merfish_stack() -> Stack:
    """retrieve MERFISH testing data from cloudfront and expose it at the module level

    Notes
    -----
    Because download takes time, this fixture runs once per session -- that is, the download is run only once.

    Returns
    -------
    Stack :
        starfish.io.Stack object containing MERFISH data
    """
    s = Stack()
    s.read('https://s3.amazonaws.com/czi.starfish.data.public/20180607/test/MERFISH/fov_001/experiment_new.json')
    return deepcopy(s)


@pytest.fixture(scope='session')
def labeled_synthetic_dataset():
    stp = synthesize.SyntheticSpotTileProvider()
    image = ImageStack.synthetic_stack(tile_data_provider=stp.tile)
    max_proj = image.max_proj(Indices.HYB, Indices.CH, Indices.Z)
    view = max_proj.reshape((1, 1, 1) + max_proj.shape)
    dots = ImageStack.from_numpy_array(view)

    def labeled_synthetic_dataset_factory():
        return deepcopy(image), deepcopy(dots), deepcopy(stp.codebook)

    return labeled_synthetic_dataset_factory
