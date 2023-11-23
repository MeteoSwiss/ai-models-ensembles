import argparse

import ai_models_fourcastnetv2.fourcastnetv2 as nvs
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
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
    for block in model.blocks:
        spectral_attention_layer = block.filter_layer.filter
        for param in spectral_attention_layer.w:
            noise = torch.randn_like(param.data) * perturbation_strength
            param.data += noise
    return model


def save_model_weights(model, path):
    """Save the model weights."""
    torch.save({'model_state': model.state_dict()}, path)


def main():
    torch.manual_seed(args.seed)
    checkpoint_path = "weights.tar"
    model = nvs.FourierNeuralOperatorNet()
    model.zero_grad()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model_weights(model, checkpoint_path, device)

    perturbation_strength = 0.01
    model = perturb_weights(model, perturbation_strength)

    save_model_weights(model, f"weights{args.seed}.tar")


if __name__ == "__main__":
    main()
