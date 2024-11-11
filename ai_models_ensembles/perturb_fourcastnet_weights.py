import argparse
import os

import ai_models_fourcastnetv2.fourcastnetv2 as nvs
import torch

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


def load_model_weights(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        # Try adding model weights as dictionary
        new_state_dict = {
            k[7:]: v for k, v in checkpoint["model_state"].items() if k[7:] != "ged"
        }
        model.load_state_dict(new_state_dict)
    except Exception:
        model.load_state_dict(checkpoint["model_state"])
    return model


def perturb_weights(model, perturbation_strength):
    """Perturb the weights of the model."""

    print(
        f"Perturbing the weights in Layer {args.layer}",
    )
    # Perturb middle layer in the nth block
    perturbed_blocks = [args.layer]
    perturbed_layers = [int(len(model.blocks[0].filter_layer.filter.w) / 2)]

    # # Use all model blocks and layers
    # perturbed_blocks = list(range(len(model.blocks)))
    # perturbed_layers = list(range(len(model.blocks[0].filter_layer.filter.w)))

    for block in [model.blocks[i] for i in perturbed_blocks]:
        layer = block.filter_layer.filter
        for param in [layer.w[i] for i in perturbed_layers]:
            noise = torch.randn_like(param.data[:, :, 0]) * perturbation_strength
            param.data[:, :, 0] += noise
        print(
            "Amplitude tensor now ranges from",
            float(torch.min(param.data[:, :, 0])),
            "to",
            float(torch.max(param.data[:, :, 0])),
            "with a mean of",
            float(torch.mean(param.data[:, :, 0])),
        )
    return model


def save_model_weights(model, path):
    """Save the model weights."""
    torch.save({"model_state": model.state_dict()}, path)


def main():
    path_out = os.path.join(
        args.out_dir,
        str(args.date_time),
        args.model_name,
        f"init_{args.perturbation_init}_latent_{args.perturbation_latent}_layer_{args.layer}",
        str(args.member),
        "weights.tar",
    )

    print(
        "Perturbing the weights in the FourierNeuralOperatorBlock by",
        args.perturbation_latent,
    )

    torch.manual_seed(args.member)
    checkpoint_path = os.path.join(
        args.out_dir, str(args.date_time), args.model_name, "weights.tar"
    )
    model = nvs.FourierNeuralOperatorNet()
    model.zero_grad()
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model_weights(model, checkpoint_path, device)
    model = perturb_weights(model, args.perturbation_latent)

    save_model_weights(model, path_out)


if __name__ == "__main__":
    main()
