import numpy as np
import pytest

from starfish.test.dataset_fixtures import labeled_synthetic_dataset
from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector


def test_create_intensity_table(labeled_synthetic_dataset):
    image, dots = labeled_synthetic_dataset()
    min_sigma = 1
    max_sigma = 10
    num_sigma = 30
    threshold = 4000
    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        blobs_stack=dots,
        measurement_type='max',
    )
    # import pdb; pdb.set_trace()
    intensities = gsd.find(hybridization_image=image)
    assert intensities.shape[0] == 20


def test_create_intensity_table_raises_value_error_when_no_spots_detected(
        labeled_synthetic_dataset):
    image, dots = labeled_synthetic_dataset()
    min_sigma = 1
    max_sigma = 10
    num_sigma = 30
    threshold = 40000  # this should cause no spots to be detected
    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        blobs_stack=dots,
        measurement_type='max',
    )
    # import pdb; pdb.set_trace()
    with pytest.raises(ValueError):
        gsd.find(hybridization_image=image)


