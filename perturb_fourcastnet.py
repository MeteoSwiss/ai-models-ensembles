import argparse
import os

import ai_models_fourcastnetv2.fourcastnetv2 as nvs
import torch

# Create an argument parser
parser = argparse.ArgumentParser(
    description="Perturb the weights in the FourierNeuralOperatorBlock.")
parser.add_argument("model_name", type=str, help="The ai-model name")
parser.add_argument(
    "date_time",
    type=str,
    help="Date and time in the format YYYYMMDDHHMM")
parser.add_argument("perturbation", type=float, help="The perturbation size")
parser.add_argument(
    "member", type=int,
    help="The ensemble member number and seed for the perturbation.")
args = parser.parse_args()


def load_model_weights(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        # Try adding model weights as dictionary
        new_state_dict = {k[7:]: v for k,
                          v in checkpoint["model_state"].items() if k[7:] != "ged"}
        model.load_state_dict(new_state_dict)
    except Exception:
        model.load_state_dict(checkpoint["model_state"])
    return model


def perturb_weights(model, perturbation_strength):
    """Perturb the weights of the model."""

    # Perturb the middle layer in the middle block
    perturbed_blocks = [int(len(model.blocks) / 2)]
    perturbed_layers = [int(len(model.blocks[0].filter_layer.filter.w) / 2)]

    for block in [model.blocks[i] for i in perturbed_blocks]:
        spectral_attention_layer = block.filter_layer.filter
        for param in [spectral_attention_layer.w[i] for i in perturbed_layers]:
            noise = torch.randn_like(param.data) * perturbation_strength * 0.1
            param.data += noise
            print("Perturbing this block:")
            print(block)
            print(
                "Perturbing SpectralAttentionS2 layer",
                perturbed_layers,
                "with shape",
                param.data.shape)
            print(
                "Tensor now ranges from", torch.min(
                    param.data), "to", torch.max(
                    param.data), "with a mean of", torch.mean(
                    param.data))
    return model


def save_model_weights(model, path):
    """Save the model weights."""
    torch.save({"model_state": model.state_dict()}, path)


def main():
    path_out = os.path.join(
        args.model_name,
        str(args.date_time),
        str(args.perturbation),
        str(args.member),
        "weights.tar")

    print(
        "Perturbing the weights in the FourierNeuralOperatorBlock by",
        args.perturbation * 0.1)

    torch.manual_seed(args.member)
    checkpoint_path = args.model_name + "/weights.tar"
    model = nvs.FourierNeuralOperatorNet()
    model.zero_grad()
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model_weights(model, checkpoint_path, device)
    model = perturb_weights(model, args.perturbation)

    save_model_weights(model, path_out)


if __name__ == "__main__":
    main()
