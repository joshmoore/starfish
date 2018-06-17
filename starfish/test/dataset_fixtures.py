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


# TODO this doesn't work yet because tony's thing isn't merged.
@pytest.fixture(scope='session')
def labeled_synthetic_dataset():
    stp = synthesize.SyntheticSpotTileProvider()
    image = ImageStack.synthetic_stack(tile_data_provider=stp.tile)
    dots = image.max_proj(Indices.CH, Indices.HYB, Indices.Z)
    stack = Stack.from_data(image, aux_dict={'dots': dots})

    def labeled_synthetic_dataset_factory():
        return deepcopy(stack)

    return labeled_synthetic_dataset_factory
