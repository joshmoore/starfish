from starfish.test.dataset_fixtures import labeled_synthetic_dataset
from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector


def test_create_intensity_table(labeled_synthetic_dataset):
    stack = labeled_synthetic_dataset()
    min_sigma = 1
    max_sigma = 10
    num_sigma = 30
    threshold = 0.01
    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        blobs_image_name='dots',
        measurement_type='mean',
    )
    # import pdb; pdb.set_trace()
    gsd.find(image_stack=stack)
