from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
from models import LowEnergyTwoModel
import glob


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_other_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall_other/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
        "wall_other": probe_val_wall_other_ds,
    }

    return probe_train_ds, probe_val_ds


def load_expert_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_expert_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_expert/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_expert_ds = {
        "expert": create_wall_dataloader(
            data_path=f"{data_path}/probe_expert/val",
            probing=True,
            device=device,
            train=False,
        )
    }

    return probe_train_expert_ds, probe_val_expert_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    #model = MockModel()
    model = LowEnergyTwoModel(device=device, repr_dim=256).to(device)
    model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    model = load_model()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)

    probe_train_expert_ds, probe_val_expert_ds = load_expert_data(device)
    evaluate_model(device, model, probe_train_expert_ds, probe_val_expert_ds)
