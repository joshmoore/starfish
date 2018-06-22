from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.test.dataset_fixtures import *


def test_spots_match_coordinates_of_simple_spot(synthetic_single_spot_imagestack):
    image = synthetic_single_spot_imagestack

    gsd = GaussianSpotDetector(
        min_sigma=0.5,
        max_sigma=10,
        num_sigma=5,
        threshold=0,
        blobs_stack=image,
        measurement_type='max',
    )
    intensities = gsd.find(hybridization_image=image)
    assert intensities.shape[0] == 1
    assert np.abs(int(intensities.y.values) - 10) < 2
    assert np.abs(int(intensities.x.values) - 90) < 2


def test_spots_match_coordinates_of_synthesized_spots(
        synthetic_dataset_with_truth_values_and_called_spots):

    codebook, true_intensities, image, dots, intensities = (
        synthetic_dataset_with_truth_values_and_called_spots)
    assert true_intensities.shape == intensities.shape

    assert sorted(true_intensities.coords['x'].values) == sorted(intensities.coords['x'].values)
    assert sorted(true_intensities.coords['y'].values) == sorted(intensities.coords['y'].values)


def test_create_intensity_table(synthetic_dataset_with_truth_values_and_called_spots):
    codebook, true_intensities, image, dots, intensities = (
        synthetic_dataset_with_truth_values_and_called_spots)

    assert intensities.shape[0] == 5


def test_create_intensity_table_raises_value_error_when_no_spots_detected(
        synthetic_dataset_with_truth_values_and_called_spots):

    codebook, true_intensities, image, dots, intensities = (
        synthetic_dataset_with_truth_values_and_called_spots)

    min_sigma = 1
    max_sigma = 10
    num_sigma = 30
    threshold = 1.0

    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        blobs_stack=dots,
        measurement_type='max',
    )
    with pytest.raises(ValueError):
        gsd.find(hybridization_image=image)


