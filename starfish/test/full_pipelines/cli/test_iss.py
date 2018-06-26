import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Sequence

import jsonpath_rw

from starfish.util import clock
import pytest


def get_jsonpath_from_file(json_filepath_components: Sequence[str], jsonpath: str):
    """
    Given a series of filepath components, join them to find a file <FILE>.  Open that file, and locate a specific value
    in the json structure <PATH>.  Join the directory path of <FILE> and <PATH> and return that.

    For example, if json_path_components is ["/tmp", "formatted", "experiment.json"] and jsonpath is
    "$['hybridization']", this method will open /tmp/formatted/experiment.json, decode that as a json document, and
    locate the value of the key 'hybridization'.  It will return /tmp/formatted/XXX, where XXX is the value of the key.
    """
    json_filepath = os.path.join(*json_filepath_components)
    dirname = os.path.dirname(json_filepath)
    with open(json_filepath, "r") as fh:
        document = json.load(fh)
        return os.path.join(dirname, jsonpath_rw.parse(jsonpath).find(document)[0].value)


@pytest.mark.skip('long test')
# @unittest.skip('long test, failing for unknown reasons (test with synthetic data!)')
class TestWithIssData(unittest.TestCase):
    SUBDIRS = (
        "raw",
        "formatted",
        "registered",
        "filtered",
        "results",
    )

    STAGES = (
        [
            sys.executable,
            "examples/get_iss_data.py",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "raw"),
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "formatted"),
            "--d", "1",
        ],
        [
            "starfish", "register",
            "--input", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [tempdir, "formatted", "experiment.json"],
                "$['hybridization_images']",
            ),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered", "hybridization.json"),
            "fourier_shift",
            "--reference-stack", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [tempdir, "formatted", "experiment.json"],
                "$['auxiliary_images']['dots']",
            ),
            "--upsampling", "1000",
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "registered", "hybridization.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "hybridization.json"),
            "white_tophat",
            "--disk-size", "15",
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [tempdir, "formatted", "experiment.json"],
                "$['auxiliary_images']['nuclei']",
            ),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "nuclei.json"),
            "white_tophat",
            "--disk-size", "15",
        ],
        [
            "starfish", "filter",
            "--input", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [tempdir, "formatted", "experiment.json"],
                "$['auxiliary_images']['dots']",
            ),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "dots.json"),
            "white_tophat",
            "--disk-size", "15",
        ],
        [
            "starfish", "detect_spots",
            "--input", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "hybridization.json"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results"),
            "gaussian_spot_detector",
            "--blobs-stack", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "dots.json"),
            "--min-sigma", "4",
            "--max-sigma", "6",
            "--num-sigma", "20",
            "--threshold", "0.01",
        ],
        [
            "starfish", "segment",
            "--hybridization-stack", lambda tempdir, *args, **kwargs: os.path.join(
                tempdir, "filtered", "hybridization.json"),
            "--nuclei-stack", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "filtered", "nuclei.json"),
            "-o", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "regions.geojson"),
            "watershed",
            "--dapi-threshold", ".16",
            "--input-threshold", ".22",
            "--min-distance", "57",
        ],
        [
            "starfish", "gene_assignment",
            "--coordinates-geojson",
            lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "regions.geojson"),
            "--spots-json", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "spots.nc"),
            "--output", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "regions.json"),
            "point_in_poly",
        ],
        [
            "starfish", "decode",
            "-i", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "spots.nc"),
            "--codebook", lambda tempdir, *args, **kwargs: get_jsonpath_from_file(
                [tempdir, "formatted", "experiment.json"],
                "$['codebook']",
            ),
            "-o", lambda tempdir, *args, **kwargs: os.path.join(tempdir, "results", "spots.nc"),
            "iss",
        ],
    )

    def test_run_pipeline(self):
        tempdir = tempfile.mkdtemp()
        coverage_enabled = "STARFISH_COVERAGE" in os.environ

        def callback(interval):
            print(" ".join(stage[:2]), " ==> {} seconds".format(interval))

        try:
            for subdir in TestWithIssData.SUBDIRS:
                os.makedirs("{tempdir}".format(
                    tempdir=os.path.join(tempdir, subdir)))
            for stage in TestWithIssData.STAGES:
                cmdline = [
                    element(tempdir=tempdir) if callable(element) else element
                    for element in stage
                ]
                if cmdline[0] == "starfish" and coverage_enabled:
                    coverage_cmdline = [
                        "coverage", "run",
                        "-p",
                        "--source", "starfish",
                        "-m", "starfish",
                    ]
                    coverage_cmdline.extend(cmdline[1:])
                    cmdline = coverage_cmdline
                with clock.timeit(callback):
                    subprocess.check_call(cmdline)
            with open(os.path.join(tempdir, "results", "decoded_table.json")) as fh:
                results = json.load(fh)

            counts = collections.defaultdict(lambda: 0)
            for record in results:
                counts[record["barcode"]] += 1
            tuples = [(count, barcode) for barcode, count in counts.items()]
            tuples.sort(reverse=True)
            self.assertEqual("AAGC", tuples[0][1])
            self.assertEqual("AGGC", tuples[1][1])
        finally:
            if os.getenv("TEST_ISS_KEEP_DATA") is None:
                shutil.rmtree(tempdir)
