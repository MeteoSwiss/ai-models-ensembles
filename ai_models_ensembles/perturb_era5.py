import argparse
import os

import numpy as np
from eccodes import (
    codes_get,
    codes_get_array,
    codes_grib_new_from_file,
    codes_release,
    codes_set_array,
    codes_write,
)

# Create an argument parser
parser = argparse.ArgumentParser(description="Perturb the t2m field in a grib file.")
parser.add_argument("out_dir", type=str, help="The output directory")
parser.add_argument(
    "date_time", type=str, help="Date and time in the format YYYYMMDDHHMM"
)
parser.add_argument("model_name", type=str, help="The ai-model name")
parser.add_argument("perturbation_init", type=float, help="The init perturbation size")
parser.add_argument(
    "perturbation_latent", type=float, help="The latent perturbation size"
)
parser.add_argument(
    "member", type=int, help="The ensemble member number and seed for the perturbation."
)
parser.add_argument(
    "layer",
    nargs="?",
    type=int,
    default=None,
    help="Optional layer index used in output path (defaults to $LAYER or 0)",
)
args = parser.parse_args()

# Open the GRIB file
f = open(
    args.out_dir + "/" + args.date_time + "/" + args.model_name + "/init_field.grib",
    "rb",
)

# Open a new GRIB file for writing
# Determine layer value (CLI > env > default 0)
layer_val = args.layer if args.layer is not None else int(os.environ.get("LAYER", 0))

path_out = os.path.join(
    args.out_dir,
    str(args.date_time),
    args.model_name,
    f"init_{args.perturbation_init}_latent_{args.perturbation_latent}_layer_{layer_val}",
    str(args.member),
    "init_field.grib",
)
print("Starting from: ", path_out)

g = open(path_out, "wb")

# Set the random numpy seed:
np.random.seed(args.member)

while True:
    # Get the next message
    gid = codes_grib_new_from_file(f)
    if gid is None:
        break

    # Get the short name of the field
    short_name = codes_get(gid, "shortName")

    # Alter the "2t" field if it's present
    # BUG: make this more general
    if short_name == "t":
        values = codes_get_array(gid, "values")
        values += np.random.normal(0, 1, size=len(values)) * args.perturbation_init
        codes_set_array(gid, "values", values)

    # Write the message to the new GRIB file
    codes_write(gid, g)

    # Delete the message
    codes_release(gid)

# Close the GRIB files
f.close()
g.close()
