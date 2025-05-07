import torch
import torch.optim as optim
import argparse
from main import get_device, evaluate_model
from main import load_expert_data, load_model, load_data
from models import LowEnergyTwoModel
from train import train_low_energy_two_model
from dataset import create_wall_dataloader


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_training_data(device):
    data_path="/scratch/DL24FA/train"

    train_ds = create_wall_dataloader(
        data_path=f"{data_path}",
        probing=False,
        device=device,
        train=True,
    )

    return train_ds


#augmentation
class HorizontalFlippedDataset:
    """Creates a horizontally flipped version of the original dataset"""
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.batch_size = original_dataset.batch_size

    def __iter__(self):
        for batch in self.dataset:
            # Flip states horizontally
            flipped_states = torch.flip(batch.states, dims=[-1])  
            
            # Flip actions (only x direction)
            flipped_actions = batch.actions.clone()
            flipped_actions[..., 0] = -flipped_actions[..., 0]  # x = -x
            
            # Create empty locations tensor with same shape as states
            empty_locations = torch.zeros_like(batch.states[:, :, 0, 0, 0])  # [B, T]
            empty_locations = empty_locations.unsqueeze(-1)  # Add feature dimension [B, T, 1]
            
            # Create new batch
            flipped_batch = type(batch)(
                states=flipped_states,
                actions=flipped_actions,
                locations=empty_locations
            )
            yield flipped_batch
    
    def __len__(self):
        return len(self.dataset)

class VerticalFlippedDataset:
    """Creates a vertically flipped version of the original dataset"""
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.batch_size = original_dataset.batch_size

    def __iter__(self):
        for batch in self.dataset:
            # Flip states vertically
            flipped_states = torch.flip(batch.states, dims=[-2])  
            
            # Flip actions (only y direction)
            flipped_actions = batch.actions.clone()
            flipped_actions[..., 1] = -flipped_actions[..., 1]  # y = -y
            
            # Create empty locations tensor with same shape as states
            empty_locations = torch.zeros_like(batch.states[:, :, 0, 0, 0])  # [B, T]
            empty_locations = empty_locations.unsqueeze(-1)  # Add feature dimension [B, T, 1]
            
            # Create new batch
            flipped_batch = type(batch)(
                states=flipped_states,
                actions=flipped_actions,
                locations=empty_locations
            )
            yield flipped_batch
    
    def __len__(self):
        return len(self.dataset)

class BothFlippedDataset:
    """Creates a horizontally and vertically flipped version of the original dataset"""
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.batch_size = original_dataset.batch_size

    def __iter__(self):
        for batch in self.dataset:
            # Flip states both horizontally and vertically
            flipped_states = torch.flip(batch.states, dims=[-2, -1])  
            
            # Flip actions in both directions
            flipped_actions = batch.actions.clone()
            flipped_actions[..., 0] = -flipped_actions[..., 0]  # x = -x
            flipped_actions[..., 1] = -flipped_actions[..., 1]  # y = -y
            
            # Create empty locations tensor with same shape as states
            empty_locations = torch.zeros_like(batch.states[:, :, 0, 0, 0])  # [B, T]
            empty_locations = empty_locations.unsqueeze(-1)  # Add feature dimension [B, T, 1]
            
            # Create new batch
            flipped_batch = type(batch)(
                states=flipped_states,
                actions=flipped_actions,
                locations=empty_locations
            )
            yield flipped_batch
    
    def __len__(self):
        return len(self.dataset)

class FullyAugmentedTrainingDataset:
    """Combines original and all flipped versions of the training dataset"""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.horizontal_dataset = HorizontalFlippedDataset(original_dataset)
        self.vertical_dataset = VerticalFlippedDataset(original_dataset)
        self.both_dataset = BothFlippedDataset(original_dataset)
        self.batch_size = original_dataset.batch_size

    def __iter__(self):
        # Yield original batches
        for batch in self.original_dataset:
            yield batch
            
        # Yield horizontally flipped batches
        for batch in self.horizontal_dataset:
            yield batch
            
        # Yield vertically flipped batches
        for batch in self.vertical_dataset:
            yield batch
            
        # Yield both flipped batches
        for batch in self.both_dataset:
            yield batch
    
    def __len__(self):
        return len(self.original_dataset) * 4  # 4x the original dataset size

def load_training_data_augmented(device):
    """Load training data including original and all flipped versions"""
    original_train_ds = load_training_data(device=device)
    augmented_train_ds = FullyAugmentedTrainingDataset(original_train_ds)
    return augmented_train_ds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train model and save .pth file")
    parser.add_argument("--plot", action="store_true", help="display plots during training")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    args = parser.parse_args()

    num_epochs = args.epochs
    train_only = args.train
    plot = args.plot
    test_mode = args.test
    learning_rate = 1.5e-4
    repr_dim = 256


    device = get_device()
    print(f'Epochs = {num_epochs}')
    print(f'Learning rate = {learning_rate}')
    print(f'Representation dimension = {repr_dim}')
    print(f'Test mode = {test_mode}')

    if train_only:
        print('Training low energy model')
        model = LowEnergyTwoModel(device=device, repr_dim=repr_dim, training=True).to(device)
        train_loader = load_training_data_augmented(device=device)

        predicted_states, target_states = train_low_energy_two_model(
            model=model,
            train_loader=train_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            test_mode=test_mode,
            plot=plot
        )
        print('Training completed')


    else:
        # evaluate the model
        print('Evaluating best_model.pth')
        model = LowEnergyTwoModel(device=device, repr_dim=256).to(device)
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))

        def count_parameters_detailed(model):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total Trainable Parameters: {total_params:,}")
            return
        count_parameters_detailed(model)
        
        probe_train_ds, probe_val_ds = load_data(device)
        evaluate_model(device, model, probe_train_ds, probe_val_ds)

        probe_train_expert_ds, probe_val_expert_ds = load_expert_data(device)
        evaluate_model(device, model, probe_train_expert_ds, probe_val_expert_ds)