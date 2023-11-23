import argparse

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
parser = argparse.ArgumentParser(description='Perturb the t2m field in a grib file.')
parser.add_argument(
    'MEMBER', type=int,
    help='The ensemble member number and seed for the perturbation.')
args = parser.parse_args()

# Open the GRIB file
f = open('../../../era5_init.grib', 'rb')

# Open a new GRIB file for writing
g = open('era5_init.grib', 'wb')

# Set the random numpy seed:
np.random.seed(args.MEMBER)

while True:
    # Get the next message
    gid = codes_grib_new_from_file(f)
    if gid is None:
        break

    # Get the short name of the field
    short_name = codes_get(gid, 'shortName')

    # Alter the "2t" field if it's present
    if short_name == '2t':
        values = codes_get_array(gid, 'values')
        values += np.random.normal(0, 1, size=len(values)) * 1e-14
        codes_set_array(gid, 'values', values)

    # Write the message to the new GRIB file
    codes_write(gid, g)

    # Delete the message
    codes_release(gid)

# Close the GRIB files
f.close()
g.close()
