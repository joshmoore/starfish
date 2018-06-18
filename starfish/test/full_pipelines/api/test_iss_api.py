from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.pipeline.filter.white_tophat import WhiteTophat
from starfish.pipeline.registration.fourier_shift import FourierShiftRegistration
import pytest
from starfish.test.dataset_fixtures import labeled_synthetic_dataset
from starfish.pipeline.features.codebook import Codebook


def test_iss_pipeline(labeled_synthetic_dataset):
    image, dots, codebook = labeled_synthetic_dataset()

    # todo the synthetic data looks weird in 3d, look into it.
    wth = WhiteTophat(disk_size=15)
    wth.filter(image)
    wth.filter(dots)

    fsr = FourierShiftRegistration(upsampling=1000, reference_stack=dots)
    fsr.register(image)

    min_sigma = 1
    max_sigma = 10
    num_sigma = 30
    threshold = 4000
    import pdb; pdb.set_trace()
    gsd = GaussianSpotDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        blobs_stack=dots,
        measurement_type='max',
    )
    intensities = gsd.find(hybridization_image=image)
    assert intensities.shape[0] == 20

    intesities = codebook.decode_per_channel_max(intensities)


