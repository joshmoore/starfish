import json
import os
import shutil
import tempfile
import warnings

import numpy as np
import pytest

from starfish.constants import Indices
from starfish.pipeline.features.codebook import Codebook
from starfish.pipeline.features.intensity_table import IntensityTable
from starfish.test.pipeline.test_intensity_table import small_intensity_table


@pytest.fixture(scope='module')
def simple_codebook_array():
    return [
        {
            Codebook.Constants.CODEWORD.value: [
                {Indices.HYB.value: 0, Indices.CH.value: 0, Codebook.Constants.VALUE.value: 1},
                {Indices.HYB.value: 1, Indices.CH.value: 1, Codebook.Constants.VALUE.value: 1}
            ],
            Codebook.Constants.GENE.value: "SCUBE2"
        },
        {
            Codebook.Constants.CODEWORD.value: [
                {Indices.HYB.value: 0, Indices.CH.value: 1, Codebook.Constants.VALUE.value: 1},
                {Indices.HYB.value: 1, Indices.CH.value: 1, Codebook.Constants.VALUE.value: 1}
            ],
            Codebook.Constants.GENE.value: "BRCA"
        },
        {
            Codebook.Constants.CODEWORD.value: [
                {Indices.HYB.value: 0, Indices.CH.value: 1, Codebook.Constants.VALUE.value: 1},
                {Indices.HYB.value: 1, Indices.CH.value: 0, Codebook.Constants.VALUE.value: 1}
            ],
            Codebook.Constants.GENE.value: "ACTB"
        }
    ]


@pytest.fixture(scope='module')
def simple_codebook_json(simple_codebook_array) -> tempfile.TemporaryFile:
    directory = tempfile.mkdtemp()
    codebook_json = os.path.join(directory, 'simple_codebook.json')
    with open(codebook_json, 'w') as f:
        json.dump(simple_codebook_array, f)

    yield codebook_json

    shutil.rmtree(directory)


def test_loading_codebook_from_json(simple_codebook_json):
    cb = Codebook.from_json(simple_codebook_json, n_ch=2, n_hyb=2)
    assert isinstance(cb, Codebook)


def test_loading_codebook_from_list(simple_codebook_array):
    cb = Codebook.from_code_array(simple_codebook_array, n_ch=2, n_hyb=2)
    assert isinstance(cb, Codebook)


def test_loading_codebook_without_specifying_ch_hyb_guesses_correct_values(simple_codebook_array):
    cb = Codebook.from_code_array(simple_codebook_array)
    assert cb.shape == (3, 2, 2)


@pytest.mark.parametrize('n_ch, n_hyb', ((2, 2), (5, 4)))
def test_loading_codebook_with_unused_channels_and_hybs(simple_codebook_json, n_ch, n_hyb):
    cb = Codebook.from_json(simple_codebook_json, n_ch=n_ch, n_hyb=n_hyb)
    assert cb.shape == (3, n_ch, n_hyb)


def test_loading_codebook_with_too_few_dims_raises_value_error(simple_codebook_json):
    with pytest.raises(ValueError):
        Codebook.from_json(simple_codebook_json, n_ch=1, n_hyb=2)

    with pytest.raises(ValueError):
        Codebook.from_json(simple_codebook_json, n_ch=2, n_hyb=1)


@pytest.fixture(scope='module')
def loaded_codebook(simple_codebook_json):
    return Codebook.from_json(simple_codebook_json, n_ch=2, n_hyb=2)


@pytest.fixture(scope='function')
def euclidean_decoded_intensities(small_intensity_table, loaded_codebook):
    decoded_intensities = loaded_codebook.decode_euclidean(small_intensity_table)
    return decoded_intensities


def test_euclidean_decode_yields_correct_output(euclidean_decoded_intensities):
    expected_gene_annotation = np.array(["ACTB", "SCUBE2", "BRCA"])
    observed_gene_annotation = euclidean_decoded_intensities[
        IntensityTable.Constants.GENE.value].values
    assert np.array_equal(expected_gene_annotation, observed_gene_annotation)


def test_indexing_on_set_genes(euclidean_decoded_intensities):
    # note that this kind of indexing produces an xarray-internal FutureWarning about float
    # conversion that we can safely ignore here.
    is_actin = euclidean_decoded_intensities[IntensityTable.Constants.GENE.value] == 'ACTB'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)

        # select only the intensities that are actin, drop the rest
        result = euclidean_decoded_intensities.where(is_actin, drop=True)

    assert result.shape == (1, 2, 2)


def test_synthetic_codes_are_on_only_once_per_channel(euclidean_decoded_intensities):
    expected_gene_annotation = np.array(["ACTB", "SCUBE2", "BRCA"])
    observed_gene_annotation = euclidean_decoded_intensities[
        IntensityTable.Constants.GENE.value].values
    assert np.array_equal(expected_gene_annotation, observed_gene_annotation)


@pytest.fixture(scope='function')
def per_channel_max_decoded_intensities(small_intensity_table, loaded_codebook):
    decoded_intensities = loaded_codebook.decode_per_channel_max(small_intensity_table)
    return decoded_intensities


def test_per_channel_max_decode_yields_expected_results(per_channel_max_decoded_intensities):
    expected_gene_annotation = np.array(["ACTB", "SCUBE2", "BRCA"])
    observed_gene_annotation = per_channel_max_decoded_intensities[
        IntensityTable.Constants.GENE.value].values
    assert np.array_equal(expected_gene_annotation, observed_gene_annotation)


def test_synthetic_one_hot_codes_produce_one_channel_per_hyb():
    cb = Codebook.synthetic_one_hot_codebook(n_hyb=6, n_channel=4, n_codes=100)
    # sum over channels: only one should be "on"
    assert np.all(cb.sum(Indices.CH.value) == 1)


def test_codebook_save(loaded_codebook):
    directory = tempfile.mkdtemp()
    filename = os.path.join(directory, 'codebook.json')
    loaded_codebook.save(filename)
    reloaded = Codebook.from_json(filename, n_hyb=2, n_ch=2)

    assert np.array_equal(loaded_codebook, reloaded)
    assert np.array_equal(loaded_codebook[Indices.CH.value], reloaded[Indices.CH.value])
    assert np.array_equal(loaded_codebook[Indices.HYB.value], reloaded[Indices.HYB.value])
    assert np.array_equal(loaded_codebook[Codebook.Constants.GENE.value].values,
                          reloaded[Codebook.Constants.GENE.value].values)
