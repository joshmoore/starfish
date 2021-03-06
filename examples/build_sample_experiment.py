import argparse
import json

from examples.support import AUX_IMAGE_NAMES, write_experiment_json
from starfish.constants import Indices
from starfish.util.argparse import FsExistsType


class StarfishIndex:
    def __call__(self, spec_json):
        try:
            spec = json.loads(spec_json)
        except json.decoder.JSONDecodeError:
            raise argparse.ArgumentTypeError("Could not parse {} into a valid index specification.".format(spec_json))

        return {
            Indices.HYB: spec.get(Indices.HYB, 1),
            Indices.CH: spec.get(Indices.CH, 1),
            Indices.Z: spec.get(Indices.Z, 1),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=FsExistsType())
    parser.add_argument(
        "--fov-count",
        type=int,
        required=True,
        help="Number of FOVs in this experiment.")
    parser.add_argument(
        "--hybridization-dimensions",
        type=StarfishIndex(),
        required=True,
        help="Dimensions for the hybridization images.  Should be a json dict, with {}, {}, and {} as the possible "
             "keys.  The value should be the shape along that dimension.  If a key is not present, the value is "
             "assumed to be 0.".format(
            Indices.HYB.value,
            Indices.CH.value,
            Indices.Z.value))
    name_arg_map = dict()
    for aux_image_name in AUX_IMAGE_NAMES:
        arg = parser.add_argument(
            "--{}-dimensions".format(aux_image_name),
            type=StarfishIndex(),
            help="Dimensions for the {} images.  Should be a json dict, with {}, {}, and {} as the possible keys.  The "
                 "value should be the shape along that dimension.  If a key is not present, the value is assumed to be "
                 "0.".format(aux_image_name, Indices.HYB.value, Indices.CH.value, Indices.Z.value))
        name_arg_map[aux_image_name] = arg.dest

    args = parser.parse_args()

    write_experiment_json(
        args.output_dir, args.fov_count, args.hybridization_dimensions,
        {
            aux_image_name: getattr(args, name_arg_map[aux_image_name])
            for aux_image_name in AUX_IMAGE_NAMES
        }
    )
