from starfish.util.synthesize import SyntheticData
from starfish.pipeline.features.spots.detector.gaussian import GaussianSpotDetector
from starfish.constants import Indices
from starfish.image import ImageStack

#todo upgrade spot detection to 3d
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
        fill_dynamic_range=False
    )

    codebook = sd.codebook()
    intensities = sd.intensities(codebook=codebook)
    spots = sd.spots(intensities=intensities)
    blobs_array = spots.max_proj(Indices.CH, Indices.HYB, Indices.Z)
    blobs = ImageStack.from_numpy_array(blobs_array.reshape(1, 1, 1, *blobs_array.shape))
    gsd = GaussianSpotDetector(
        min_sigma=1, max_sigma=10, num_sigma=30, threshold=0, blobs_stack=blobs)
    # todo I think something here is wrong with 3d spot detection
    calculated_intensities = gsd.find(spots)
    codebook.decode_euclidean(calculated_intensities)
    import pdb; pdb.set_trace()


def test_round_trip_synthetic_data_2d():
    sd = SyntheticData(
        n_spots=1,
        n_z=1,
        n_codes=10,
        n_photons_background=0,
        background_electrons=0,
        camera_detection_efficiency=1.0,
        gray_level=1,
        ad_conversion_bits=16,
        point_spread_function=(2, 2, 2),
        fill_dynamic_range=False
    )

    codebook = sd.codebook()
    intensities = sd.intensities(codebook=codebook)
    spots = sd.spots(intensities=intensities)
    blobs_array = spots.max_proj(Indices.CH, Indices.HYB, Indices.Z)
    blobs = ImageStack.from_numpy_array(blobs_array.reshape(1, 1, 1, *blobs_array.shape))
    gsd = GaussianSpotDetector(
        min_sigma=1, max_sigma=10, num_sigma=30, threshold=0, blobs_stack=blobs)
    # todo I think something here is wrong with 3d spot detection
    calculated_intensities = gsd.find(spots)
    codebook.decode_euclidean(calculated_intensities)
    import pdb; pdb.set_trace()
