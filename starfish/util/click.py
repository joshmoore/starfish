import json
from functools import update_wrapper

import click
from click.globals import get_current_context

from starfish.types import Indices


class StarfishIndex(click.ParamType):

    name = "starfish-index"

    def convert(self, spec_json, param, ctx):
        try:
            spec = json.loads(spec_json)
        except json.decoder.JSONDecodeError:
            self.fail(
                "Could not parse {} into a valid index specification.".format(spec_json))

        return {
            str(Indices.ROUND): spec.get(Indices.ROUND, 1),
            str(Indices.CH): spec.get(Indices.CH, 1),
            str(Indices.Z): spec.get(Indices.Z, 1),
        }


def dimensions_option(name, required):
    return click.option(
        "--{}-dimensions".format(name),
        type=StarfishIndex(), required=required,
        help="Dimensions for the {} images.  Should be a json dict, with {}, {}, "
             "and {} as the possible keys.  The value should be the shape along that "
             "dimension.  If a key is not present, the value is assumed to be 0."
             .format(
             name,
             Indices.ROUND.value,
             Indices.CH.value,
             Indices.Z.value))


def pass_context_and_record(f):
    def new_func(*args, **kwargs):
        ctx = get_current_context()
        record = ctx.obj.get("record", None)
        if record is not None:
            data = {ctx.info_name: {"params": ctx.params}}
            record.append(data)
        return f(ctx, *args, **kwargs)
    return update_wrapper(new_func, f)
