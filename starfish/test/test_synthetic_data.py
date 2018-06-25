import numpy as np

from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.util.synthesize import SyntheticData


def test_round_trip_synthetic_data():
    sd = SyntheticData(
        n_spots=1,
        n_codes=10,
        n_photons_background=0,
        background_electrons=0,
        camera_detection_efficiency=1.0,
        gray_level=1,
        ad_conversion_bits=16,
        point_spread_function=(2, 2, 2),
    )

    codebook = sd.codebook()
    intensities = sd.intensities(codebook=codebook)
    spots = sd.spots(intensities=intensities)
    gsd = GaussianSpotDetector(
        min_sigma=1, max_sigma=4, num_sigma=5, threshold=0, blobs_stack=spots)
    calculated_intensities = gsd.find(spots)
    codebook.decode_euclidean(calculated_intensities)

    # applying the gaussian blur to the intensities causes them to be reduced in magnitude, so
    # they won't be the same size, but they should be in the same place, and decode the same
    # way
    spot1, ch1, hyb1 = np.where(intensities.values)
    spot2, ch2, hyb2 = np.where(calculated_intensities.values)
    assert np.array_equal(spot1, spot2)
    assert np.array_equal(ch1, ch2)
    assert np.array_equal(hyb1, hyb2)
    assert np.array_equal(
        intensities.coords[intensities.Constants.GENE],
        calculated_intensities.coords[intensities.Constants.GENE]
    )
