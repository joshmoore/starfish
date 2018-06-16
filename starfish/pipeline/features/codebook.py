import functools
from itertools import product
import json
from typing import Optional, Sequence, Dict, Tuple
from typing import Union, Iterable, List, Mapping, Any

import numpy as np
import pandas as pd
import xarray as xr

from starfish.constants import Indices, AugmentedEnum
from starfish.pipeline.features.intensity_table import IntensityTable


class Codebook(xr.DataArray):
    """Codebook for a spatial transcriptomics experiment

    The codebook is a three dimensional tensor whose values are the expected intensity of a spot
    for each code in each hybridization round and each color channel. This class supports the
    construction of synthetic codebooks for testing, and exposes decode methods to assign gene
    identifiers to spots. This codebook provides an in-memory representation of the codebook
    defined in the spaceTx format.

    The codebook is a subclass of xarray, and exposes the complete public API of that package in
    addition to the methods and constructors listed below.

    Constructors
    ------------
    from_code_array(code_array, n_hyb, n_ch)
        construct a codebook from an array of codewords
    from_json(json_codebook, n_hyb, n_ch)
        construct a codebook from a spaceTx spec-compliant json codebook
    synthetic_one_hot_codebook
        construct a codebook of raondom codes where only one channel is on per hybridization round

    Methods
    -------
    decode_euclidean(intensities)
        find the closest code for each spot in intensities by euclidean distance
    decode_per_channel_maximum(intensities)
        find codes that match the per-channel max intensity for each spot in intensities

    See Also
    --------
    <link to spaceTx format>

    """

    class Constants(AugmentedEnum):
        CODEWORD = 'codeword'
        GENE = 'gene_name'
        VALUE = 'v'

    def __init__(
            self, data: Union[pd.DataFrame, np.ndarray, xr.DataArray],
            coords: Iterable[Union[pd.Index, pd.MultiIndex]],
            *args: Tuple, **kwargs: Dict
    ) -> None:
        """xarray class constructor

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray, xr.DataArray]
        coords : Iterable[Union[pd.Index, pd.MultiIndex]]
        args : Tuple
        kwargs : Dict
        """
        super().__init__(data, coords, *args, **kwargs)

    @classmethod
    def from_code_array(
            cls, code_array: List[Mapping[str, Any]],
            n_hyb: Optional[int]=None, n_ch: Optional[int]=None) -> "Codebook":
        """

        Parameters
        ----------
        code_array
        n_hyb : Optional[int]
        n_ch : Optional[int]

        Returns
        -------

        """

        # guess the max hyb and channel if not provided, otherwise check provided values are valid
        max_hyb, max_ch = 0, 0

        for code in code_array:
            for entry in code[Codebook.Constants.CODEWORD.value]:
                max_hyb = max(max_hyb, entry[Indices.HYB])
                max_ch = max(max_ch, entry[Indices.CH])

        # set n_ch and n_hyb if either were not provided
        n_hyb = n_hyb if n_hyb is not None else max_hyb + 1
        n_ch = n_ch if n_ch is not None else max_ch + 1

        # raise errors if provided n_hyb or n_ch are out of range
        if max_hyb + 1 > n_hyb:
            raise ValueError(
                f'code detected that requires a hybridization value ({max_hyb}) that is greater '
                f'than provided n_hyb: {n_hyb}')
        if max_ch + 1 > n_ch:
            raise ValueError(
                f'code detected that requires a hybridization value ({max_hyb}) that is greater '
                f'than provided n_hyb: {n_hyb}')

        for code in code_array:

            if not isinstance(code, dict):
                raise ValueError(f'codebook must be an array of dictionary codes. Found: {code}.')

            # verify all necessary fields are present
            required_fields = {Codebook.Constants.CODEWORD.value, Codebook.Constants.GENE.value}
            missing_fields = required_fields.difference(code)
            if missing_fields:
                raise ValueError(
                    f'Each entry of codebook must contain {required_fields}. Missing fields: '
                    f'{missing_fields}')

        # empty codebook
        code_data = cls(
            data=np.zeros((len(code_array), n_ch, n_hyb), dtype=np.uint8),
            coords=(
                pd.Index(
                    [d[Codebook.Constants.GENE.value] for d in code_array],
                    name=Codebook.Constants.GENE.value
                ),
                pd.Index(np.arange(n_ch), name=Indices.CH.value),
                pd.Index(np.arange(n_hyb), name=Indices.HYB.value),
            )
        )

        # fill the codebook
        for code_dict in code_array:
            codeword = code_dict[Codebook.Constants.CODEWORD.value]
            gene = code_dict[Codebook.Constants.GENE.value]
            for entry in codeword:
                code_data.loc[gene, entry[Indices.CH.value], entry[Indices.HYB.value]] = entry[
                    Codebook.Constants.VALUE.value]

        return code_data

    @classmethod
    def from_json(cls, json_codebook, n_hyb, n_ch) -> "Codebook":
        """

        Parameters
        ----------
        json_codebook
        n_hyb
        n_ch

        Returns
        -------

        """
        with open(json_codebook, 'r') as f:
            code_array = json.load(f)
        return cls.from_code_array(code_array, n_hyb, n_ch)

    @staticmethod
    def append_multiindex_level(multiindex, data, name):
        """stupid thing necessary because pandas doesn't do this"""
        frame = multiindex.to_frame()
        frame[name] = data
        frame.set_index(name, append=True, inplace=True)
        return frame.index

    def decode_euclidean(self, intensities: IntensityTable) -> IntensityTable:

        def min_euclidean_distance(observation, codes) -> np.ndarray:
            squared_diff = (codes - observation) ** 2
            code_distances = np.sqrt(squared_diff.sum((Indices.CH, Indices.HYB)))
            # order of codes changes here (automated sorting on the reshaping?)
            return code_distances

        norm_intensities = intensities.groupby(IntensityTable.Indices.FEATURES.value).apply(
            lambda x: x / x.sum())
        norm_codes = self.groupby(Codebook.Constants.GENE.value).apply(lambda x: x / x.sum())

        func = functools.partial(min_euclidean_distance, codes=norm_codes)
        distances = norm_intensities.groupby(IntensityTable.Indices.FEATURES.value).apply(func)

        qualities = 1 - distances.min(Codebook.Constants.GENE.value)
        closest_code_index = distances.argmin(Codebook.Constants.GENE.value)
        gene_ids = distances.indexes[
            Codebook.Constants.GENE.value].values[closest_code_index.values]
        with_genes = self.append_multiindex_level(
            intensities.indexes[IntensityTable.Indices.FEATURES.value], gene_ids, 'gene')
        with_qualities = self.append_multiindex_level(with_genes, qualities, 'quality')

        result = IntensityTable(
            intensities=intensities,
            dims=(IntensityTable.Indices.FEATURES.value, Indices.CH.value, Indices.HYB.value),
            coords=(
                with_qualities,
                intensities.indexes[Indices.CH.value],
                intensities.indexes[Indices.HYB.value]
            )
        )
        return result

    def decode_per_channel_max(self, intensities) -> IntensityTable:

        def view_row_as_element(array) -> np.ndarray:
            nrows, ncols = array.shape
            dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                     'formats': ncols * [array.dtype]}
            return array.view(dtype)

        max_channels = intensities.argmax(Indices.CH.value)
        codes = self.argmax(Indices.CH.value)

        a = view_row_as_element(codes.values.reshape(self.shape[0], -1))
        b = view_row_as_element(max_channels.values.reshape(intensities.shape[0], -1))

        genes = np.empty(intensities.shape[0], dtype=object)
        genes.fill('None')

        for i in np.arange(a.shape[0]):
            genes[np.where(a[i] == b)[0]] = codes['gene_name'][i]
        # TODO replace me with self['gene'] = ('features', genes)
        with_genes = self.append_multiindex_level(
            intensities.indexes[IntensityTable.Indices.FEATURES.value],
            genes.astype('U'),
            'gene')

        return IntensityTable(
            intensities=intensities,
            dims=(IntensityTable.Indices.FEATURES.value, Indices.CH.value, Indices.HYB.value),
            coords=(
                with_genes,
                intensities.indexes[Indices.CH.value],
                intensities.indexes[Indices.HYB.value]
            )
        )

    @classmethod
    def synthetic_one_hot_codebook(
            cls, n_hyb: int, n_channel: int, n_codes: int, gene_names: Optional[Sequence]=None
    ) -> "Codebook":
        """Generate codes where one channel is "on" in each hybridization round

        Parameters
        ----------
        n_hyb : int
            number of hybridization rounds per code
        n_channel : int
            number of channels per code
        n_codes : int
            number of codes to generate
        gene_names : Optional[List[str]]
            if provided, names for genes in codebook

        Returns
        -------
        List[Dict] :
            list of codewords

        """

        # TODO clean up this code, generate Codebooks directly
        # construct codes
        # this can be slow when n_codes is large and n_codes ~= n_possible_codes
        codes = set()
        while len(codes) < n_codes:
            codes.add(tuple([np.random.randint(0, n_channel) for _ in np.arange(n_hyb)]))

        # construct codewords from code
        codewords = [
            [
                {Indices.HYB.value: h, Indices.CH.value: c, 'v': 1} for h, c in enumerate(code)
            ] for code in codes
        ]

        # make a codebook from codewords
        if gene_names is None:
            gene_names = np.arange(n_codes)
        assert n_codes == len(gene_names)

        codebook = [{Codebook.Constants.CODEWORD.value: w, Codebook.Constants.GENE.value: g}
                    for w, g in zip(codewords, gene_names)]

        return cls.from_code_array(codebook, n_hyb=n_hyb, n_ch=n_channel)

