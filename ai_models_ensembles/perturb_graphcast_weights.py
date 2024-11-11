import argparse
import os

import numpy as np

# Create an argument parser
parser = argparse.ArgumentParser(
    description="Perturb the weights in the FourierNeuralOperatorBlock."
)
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
parser.add_argument("layer", type=int, help="The layer to perturb")
args = parser.parse_args()


def perturb_weights(params, perturbation_strength):
    """Perturb the weights of layer X in the processor of the GraphCast model."""

    keys_to_perturb = [
        # f"params:mesh_gnn/~_networks_builder/processor_edges_{args.layer}_mesh_mlp/~/linear_0:w",
        f"params:mesh_gnn/~_networks_builder/processor_edges_{args.layer}_mesh_mlp/~/linear_1:w",
    ]

    for key in keys_to_perturb:
        if key in params:
            param = params[key]
            noise = np.random.randn(*param.shape) * args.perturbation_latent
            params[key] = param + noise

            print(f"Perturbed {key}")
            print(f"  Range: [{params[key].min():.4f}, {params[key].max():.4f}]")
            print(f"  Mean: {params[key].mean():.4f}")
        else:
            print(f"Warning: Key {key} not found in params")

    return params


def main():
    name = "params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"
    checkpoint_path = os.path.join(
        args.out_dir, str(args.date_time), args.model_name, name
    )
    params = dict(np.load(checkpoint_path))

    np.random.seed(args.member)

    print(
        f"Perturbing the weights in layer {args.layer} of the processor by {args.perturbation_latent}"
    )

    params = perturb_weights(params, args.perturbation_latent)

    # Save the perturbed weights
    path_out = os.path.join(
        args.out_dir,
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}_layer_{args.layer}",
        str(args.member),
        name,
    )

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    np.savez(path_out, **params)
    print(f"Saved perturbed weights to {path_out}")


if __name__ == "__main__":
    main()
