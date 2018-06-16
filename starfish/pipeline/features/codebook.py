import functools
import json
from typing import Any, Dict, List, Optional, Sequence

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
        construct a codebook from a spaceTx-spec array of codewords
    from_json(json_codebook, n_hyb, n_ch)
        load a codebook from a spaceTx spec-compliant json file
    synthetic_one_hot_codebook
        construct a codebook of random codes where only one channel is on per hybridization round

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

    @classmethod
    def _empty_codebook(cls, code_names: Sequence[str], n_ch: int, n_hyb: int):
        """create an empty codebook of shape (code_names, n_ch, n_hyb)

        Parameters
        ----------
        code_names : Sequence[str]
            the genes to be coded
        n_ch : int
            number of channels used to build the codes
        n_hyb : int
            number of hybridization rounds used to build the codes

        Returns
        -------
        Codebook :
            codebook whose values are all zero

        """
        codes_index = pd.Index(code_names, name=Codebook.Constants.GENE.value)
        return cls(
            data=np.zeros((codes_index.shape[0], n_ch, n_hyb), dtype=np.uint8),
            coords=(
                codes_index,
                pd.Index(np.arange(n_ch), name=Indices.CH.value),
                pd.Index(np.arange(n_hyb), name=Indices.HYB.value),
            )
        )

    @classmethod
    def from_code_array(
            cls, code_array: List[Dict[str, Any]],
            n_hyb: Optional[int]=None, n_ch: Optional[int]=None) -> "Codebook":
        """ construct a codebook from a spaceTx-spec array of codewords

        Parameters
        ----------
        code_array : List[Dict[str, Any]]
            Array of dictionaries, each containing a codeword and gene_name
        n_hyb : Optional[int]
            The number of hybridization rounds used in the codes. Will be inferred if not provided
        n_ch : Optional[int]
            The number of channels used in the codes. Will be inferred if not provided

        Returns
        -------
        Codebook :
            Codebook with shape (genes, channels, hybridization_rounds)

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

        # verify codebook structure and fields
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

        gene_names = [w[Codebook.Constants.GENE.value] for w in code_array]
        code_data = cls._empty_codebook(gene_names, n_ch, n_hyb)

        # fill the codebook
        for code_dict in code_array:
            codeword = code_dict[Codebook.Constants.CODEWORD.value]
            gene = code_dict[Codebook.Constants.GENE.value]
            for entry in codeword:
                code_data.loc[gene, entry[Indices.CH.value], entry[Indices.HYB.value]] = entry[
                    Codebook.Constants.VALUE.value]

        return code_data

    @classmethod
    def from_json(cls, json_codebook: str, n_hyb: Optional[int], n_ch: Optional[int]) -> "Codebook":
        """Load a codebook from a spaceTx spec-compliant json file

        Parameters
        ----------
        json_codebook : str
            path to json file containing a spaceTx codebook
        n_hyb : Optional[int]
            The number of hybridization rounds used in the codes. Will be inferred if not provided
        n_ch : Optional[int]
            The number of channels used in the codes. Will be inferred if not provided

        Returns
        -------
        Codebook :
            Codebook with shape (genes, channels, hybridization_rounds)

        """
        with open(json_codebook, 'r') as f:
            code_array = json.load(f)
        return cls.from_code_array(code_array, n_hyb, n_ch)

    def save(self, filename: str) -> None:
        """save a codebook to json

        Notes
        -----
        This enforces the following typing of codebooks:
        ch, hyb: int
        value: float
        gene: str

        Parameters
        ----------
        filename : str
            filename

        """
        code_array = []
        for gene in self[self.Constants.GENE.value]:
            codeword = []
            for ch in self[Indices.CH.value]:
                for hyb in self[Indices.HYB.value]:
                    if self.loc[gene, ch, hyb]:
                        codeword.append(
                            {
                                Indices.CH.value: int(ch),
                                Indices.HYB.value: int(hyb),
                                self.Constants.VALUE.value: float(self.loc[gene, ch, hyb])
                            })
            code_array.append({
                self.Constants.CODEWORD.value: codeword,
                self.Constants.GENE.value: str(gene.values)
            })

        with open(filename, 'w') as f:
            json.dump(code_array, f)

    def decode_euclidean(self, intensities: IntensityTable) -> IntensityTable:
        """Assign the closest gene by euclidean distance to each feature in an intensity table

        Parameters
        ----------
        intensities : IntensityTable
            features to be decoded

        Returns
        -------
        IntensityTable :
            intensity table containing additional data variables for gene assignments and feature
            qualities

        """

        def _min_euclidean_distance(observation: xr.DataArray, codes: Codebook) -> np.ndarray:
            """find the code with the closest euclidean distance to observation

            Parameters
            ----------
            observation : xr.DataArray
                2-dimensional DataArray of shape (n_ch, n_hyb)
            codes :
                Codebook containing codes to compare to observation

            Returns
            -------
            np.ndarray :
                1-d vector containing the distance of each code to observation

            """
            squared_diff = (codes - observation) ** 2
            code_distances = np.sqrt(squared_diff.sum((Indices.CH, Indices.HYB)))
            # order of codes changes here (automated sorting on the reshaping?)
            return code_distances

        # normalize both the intensities and the codebook
        norm_intensities = intensities.groupby(IntensityTable.Constants.FEATURES.value).apply(
            lambda x: x / x.sum())
        norm_codes = self.groupby(Codebook.Constants.GENE.value).apply(lambda x: x / x.sum())

        # calculate pairwise euclidean distance between codes and features
        func = functools.partial(_min_euclidean_distance, codes=norm_codes)
        distances = norm_intensities.groupby(IntensityTable.Constants.FEATURES.value).apply(func)

        # calculate quality of each decoded spot
        qualities = 1 - distances.min(Codebook.Constants.GENE.value)
        qualities_index = pd.Index(qualities)

        # identify genes associated with closest codes
        closest_code_index = distances.argmin(Codebook.Constants.GENE.value)
        gene_ids = distances.indexes[
            Codebook.Constants.GENE.value].values[closest_code_index.values]
        gene_index = pd.Index(gene_ids)

        # set new values on the intensity table in-place
        intensities[IntensityTable.Constants.GENE.value] = (
            IntensityTable.Constants.FEATURES.value, gene_index)
        intensities[IntensityTable.Constants.QUALITY.value] = (
            IntensityTable.Constants.FEATURES.value, qualities_index)

        return intensities

    def decode_per_channel_max(self, intensities: IntensityTable) -> IntensityTable:
        """decode features by comparing the per-channel max value of each feature

        Notes
        -----
        If no code matches the per-channel max of a feature, it will be assigned np.nan instead
        of a gene value

        Parameters
        ----------
        intensities : IntensityTable
            features to be decoded

        Returns
        -------
        IntensityTable :
            intensity table containing additional data variables for gene assignments

        """

        def _view_row_as_element(array: np.ndarray) -> np.ndarray:
            """view an entire code as a single element

            This view allows vectors (codes) to be compared for equality without need for multiple
            comparisons by casting the data in each code to a structured dtype that registers as
            a single value

            Parameters
            ----------
            array : np.ndarray
                2-dimensional numpy array of shape (n_observations, (n_ch * n_hyb)) where
                observations may be either features or codes.

            Returns
            -------
            np.ndarray :
                1-dimensional vector of shape n_observations

            """
            nrows, ncols = array.shape
            dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                     'formats': ncols * [array.dtype]}
            return array.view(dtype)

        max_channels = intensities.argmax(Indices.CH.value)
        codes = self.argmax(Indices.CH.value)

        a = _view_row_as_element(codes.values.reshape(self.shape[0], -1))
        b = _view_row_as_element(max_channels.values.reshape(intensities.shape[0], -1))

        genes = np.empty(intensities.shape[0], dtype=object)
        genes.fill(np.nan)

        for i in np.arange(a.shape[0]):
            genes[np.where(a[i] == b)[0]] = codes['gene_name'][i]
        gene_index = pd.Index(genes)

        intensities[IntensityTable.Constants.GENE.value] = (
            IntensityTable.Constants.FEATURES.value, gene_index)

        return intensities

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

        # TODO clean up this code, generate Codebooks directly using _empty_codebook
        # construct codes
        # this can be slow when n_codes is large and n_codes ~= n_possible_codes
        codes = set()
        while len(codes) < n_codes:
            codes.add(tuple([np.random.randint(0, n_channel) for _ in np.arange(n_hyb)]))

        # construct codewords from code
        codewords = [
            [
                {
                    Indices.HYB.value: h, Indices.CH.value: c, 'v': 1
                } for h, c in enumerate(code)
            ] for code in codes
        ]

        # make a codebook from codewords
        if gene_names is None:
            gene_names = np.arange(n_codes)
        assert n_codes == len(gene_names)

        codebook = [{Codebook.Constants.CODEWORD.value: w, Codebook.Constants.GENE.value: g}
                    for w, g in zip(codewords, gene_names)]

        return cls.from_code_array(codebook, n_hyb=n_hyb, n_ch=n_channel)

