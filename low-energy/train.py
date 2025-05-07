import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from output import plot_state_norms, plot_gradient_norms, collect_gradient_norms


def get_subsequences(data, seq_len):
    """
    Generates all possible subsequences of length seq_len from the input data.
    Args:
        data (Tensor): Input tensor of shape (B, T, ...).
        seq_len (int): Desired sequence length.
    Returns:
        Tensor: Subsequence tensor of shape (B * num_slices, seq_len, ...).
    """
    B, T = data.shape[:2]
    num_slices = T - seq_len + 1
    slices = []
    for b in range(B):
        for i in range(num_slices):
            slices.append(data[b, i:i+seq_len])
    new_data = torch.stack(slices, dim=0)
    return new_data  # shape (B * num_slices, seq_len, ...)


def train_low_energy_two_model(
    model, 
    train_loader, 
    num_epochs=50, 
    learning_rate=1e-4, 
    device="cuda", 
    test_mode=False, 
    plot=False,
):
    optimizer = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.predictor.parameters()},
        {'params': model.wall_encoder.parameters()}
    ], lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6, 
        verbose=True
    )
    
    progress_bar = tqdm(range(num_epochs * len(train_loader)))
    
    sequence_lengths = [3, 9, 17]
    change_points = [0, 3, 6]

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        model.train()
        epoch_loss = 0.0
        count = 0

        if plot:
            gradient_norms = {"encoder": [], "target_encoder": [], "predictor": []}

        current_seq_len_states = sequence_lengths[-1]
        for change_point, seq_len in zip(change_points, sequence_lengths):
            if epoch >= change_point:
                current_seq_len_states = seq_len
            else:
                break
        seq_len_states = current_seq_len_states
        seq_len_actions = seq_len_states - 1

        # Training loop
        
        for batch in train_loader:
            batch_size = batch.states.size(0)
            subsample_size = batch_size // 4
            indices = torch.randperm(batch_size)[:subsample_size]
            states = batch.states[indices].to(device)
            actions = batch.actions[indices].to(device)

            states_subseq = get_subsequences(states, seq_len_states)
            actions_subseq = get_subsequences(actions, seq_len_actions)

            batch_size = states_subseq.shape[0]
            mini_batch_size = 128

            for i in range(0, batch_size, mini_batch_size):
                states_mini = states_subseq[i:i+mini_batch_size]
                actions_mini = actions_subseq[i:i+mini_batch_size]

                model.training = True
                predicted_states, target_states = model(states_mini, actions_mini)
                loss = model.loss(predicted_states, target_states)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            if plot:
                norms = collect_gradient_norms(model)
                for norm in norms:
                    gradient_norms[norm].append(norms[norm])

            count = count + 1
            if test_mode and count > 10:
                return predicted_states, target_states
            if plot and count%200 == 0:
               plot_state_norms(predicted_states, target_states)
               plot_gradient_norms(gradient_norms)
            
            progress_bar.update(1)

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.10f}")
        
        scheduler.step(avg_train_loss)

        torch.save(model.state_dict(), "best_model.pth")

    return predicted_states, target_states