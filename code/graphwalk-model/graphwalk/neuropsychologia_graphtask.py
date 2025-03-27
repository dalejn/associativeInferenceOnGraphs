import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
from tqdm import tqdm
import pandas as pd
from collections import Counter
import torch.nn.functional as F
import pickle
import os
import re
from scipy import stats
import itertools
from scipy.stats import ranksums
import json

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")


def create_paired_sampling(total_items=36, draw_count=18, pair_of_pairs_count=6, seed=None):
    """
    Create pairs of pairs where each pair of pairs has a shared item.

    Parameters:
    - total_items: Total number of unique items to draw from
    - draw_count: Number of items to randomly draw
    - pair_of_pairs_count: Number of pair of pairs to create
    - seed: Random seed for reproducibility (optional)

    Returns:
    - selected_pairs: List of pair of pairs, where each pair of pairs has a shared item
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Randomly draw unique items
    selected_items = np.random.choice(total_items, draw_count, replace=False).tolist()

    # Create pairs of pairs
    selected_pairs = []

    for _ in range(pair_of_pairs_count):
        # Randomly select a shared item from the drawn items
        shared_item_index = np.random.randint(len(selected_items))
        shared_item = selected_items.pop(shared_item_index)

        # Select another item for the first pair
        first_remaining_index = np.random.randint(len(selected_items))
        first_remaining = selected_items.pop(first_remaining_index)
        first_pair = [shared_item, first_remaining]

        # Select another item for the second pair
        second_remaining_index = np.random.randint(len(selected_items))
        second_remaining = selected_items.pop(second_remaining_index)
        second_pair = [shared_item, second_remaining]

        # Add the pair of pairs to the result
        selected_pairs.append((first_pair, second_pair))

    unnested = [item for pair_of_pair in selected_pairs for pair in pair_of_pair for item in pair]
    assert len(set(unnested)) == 18

    return selected_pairs


def generate_complex_pairs_distant(pair_of_pairs, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Select 3 pairs of pairs for blocked mode
    blocked_indices = np.random.choice(len(pair_of_pairs), 3, replace=False)
    blocked_pairs_of_pairs = [pair_of_pairs[i] for i in blocked_indices]
    interleaved_pairs_of_pairs = [pair_of_pairs[i] for i in range(len(pair_of_pairs)) if i not in blocked_indices]

    def randomize_pair_order(pair):
        # Randomly shuffle order within each pair
        return pair if random.random() < 0.5 else pair[::-1]

    def generate_blocked_pairs(blocked_pairs_of_pairs, triad_numbers):
        # Collect all first pairs and all second pairs separately
        first_pairs = []
        second_pairs = []

        for (first_pair, second_pair), triad_number in zip(blocked_pairs_of_pairs, triad_numbers):
            # Randomize within pairs
            first_pair_randomized = randomize_pair_order(first_pair)
            second_pair_randomized = randomize_pair_order(second_pair)

            # Add 30 instances of each pair
            first_pairs.extend([(first_pair_randomized, triad_number)] * 30)
            second_pairs.extend([(second_pair_randomized, triad_number)] * 30)

        # Shuffle first and second pairs separately
        random.shuffle(first_pairs)
        random.shuffle(second_pairs)

        # Combine first pairs in first half, second pairs in second half
        return first_pairs + second_pairs

    def generate_interleaved_pairs(interleaved_pairs_of_pairs, triad_numbers):
        # Collect all first and second pairs for interleaved trials
        all_pairs = []

        for (first_pair, second_pair), triad_number in zip(interleaved_pairs_of_pairs, triad_numbers):
            # Randomize within pairs
            first_pair_randomized = randomize_pair_order(first_pair)
            second_pair_randomized = randomize_pair_order(second_pair)

            # Add 30 instances of each pair
            all_pairs.extend([(first_pair_randomized, triad_number)] * 30)
            all_pairs.extend([(second_pair_randomized, triad_number)] * 30)

        # Shuffle all interleaved pairs
        random.shuffle(all_pairs)

        return all_pairs

    # Generate blocked pairs
    blocked_expanded_pairs = generate_blocked_pairs(
        [bpp for bpp in blocked_pairs_of_pairs],
        [i + 1 for i in range(len(blocked_pairs_of_pairs))]
    )

    # Generate interleaved pairs
    interleaved_expanded_pairs = generate_interleaved_pairs(
        [ipp for ipp in interleaved_pairs_of_pairs],
        [i + 4 for i in range(len(interleaved_pairs_of_pairs))]
    )

    # Ensure total length is 360
    assert len(blocked_expanded_pairs) + len(interleaved_expanded_pairs) == 360, "Total pairs must be 360"

    # Combine blocked and interleaved pairs
    final_pairs = []
    for b, i in zip(blocked_expanded_pairs, interleaved_expanded_pairs):
        final_pairs.extend([b, i])

    # Ensure no consecutive pairs share items
    def check_consecutive_pairs(pairs):
        for (a, _), (c, _) in zip(pairs, pairs[1:]):
            if len(set(a)) < 2:
                return False
        return True

    # Retry if consecutive pairs share items (with a maximum of 10 attempts)
    for _ in range(10):
        if check_consecutive_pairs(final_pairs):
            # Extract the specific complex pairs
            complex_pairs = final_pairs
            blocked_complex_pairs = [(pair, triad) for (pair, triad) in complex_pairs if triad in [1, 2, 3]]
            interleaved_complex_pairs = [(pair, triad) for (pair, triad) in complex_pairs if triad in [4, 5, 6]]

            return complex_pairs, blocked_complex_pairs, interleaved_complex_pairs

    raise ValueError("Could not generate pairs without consecutive shared items")


def count_total_appearances(pairs, original_pairs):
    flattened = [tuple(sorted(pair)) for pair, _ in pairs]
    original_flattened = [tuple(sorted(pair)) for pop in original_pairs for pair in pop]

    appearance_count = {}
    for pop in original_pairs:
        for pair in pop:
            sorted_pair = tuple(sorted(pair))
            count = flattened.count(sorted_pair)
            print(f"{sorted_pair}: {count} appearances")
            appearance_count[sorted_pair] = count

    return appearance_count


def find_average_index_order(complex_pairs):
    # Calculate the average index and range of indices for each unique pair
    index_map = {}
    for idx, (pair, _) in enumerate(complex_pairs):
        sorted_pair = tuple(sorted(pair))  # Ensure pairs are tuples
        if sorted_pair not in index_map:
            index_map[sorted_pair] = []
        index_map[sorted_pair].append(idx)

    avg_indices = {pair: np.mean(indices) for pair, indices in index_map.items()}
    ranges = {pair: max(indices) - min(indices) for pair, indices in index_map.items()}

    ordered_pairs = sorted(avg_indices.keys(), key=lambda pair: avg_indices[pair])
    return ordered_pairs, ranges


def visualize_pair_sequence(complex_pairs, pair_of_pairs):
    plt.figure(figsize=(10, 8))  # Reduced width from 20 to 12

    # Find ordered pairs and their ranges
    average_ordered_pairs, ranges = find_average_index_order(complex_pairs)

    # Create a color map for triads
    triad_colors = {
        1: 'red', 2: 'blue', 3: 'orange',  # Blocked triads
        4: 'purple', 5: 'green', 6: 'gold'   # Interleaved triads
    }

    # Create y-axis mapping for each pair
    pair_to_y = {pair: i for i, pair in enumerate(average_ordered_pairs)}

    # Plot each pair's occurrence
    for x, (pair, triad_number) in enumerate(complex_pairs):
        sorted_pair = tuple(sorted(pair))
        y = pair_to_y[sorted_pair]
        range_span = ranges[sorted_pair]

        # Determine marker style and color based on range
        if triad_number > 3:
            marker_style = 'o'  # Filled marker for interleaved
            facecolor = triad_colors[triad_number]
            edgecolor = triad_colors[triad_number]
        else:
            marker_style = 'o'  # Unfilled marker for blocked
            facecolor = 'none'
            edgecolor = triad_colors[triad_number]

        plt.scatter(x, y, color=facecolor, edgecolors=edgecolor, s=20, marker=marker_style)  # Reduced point size from 50 to 20

    plt.xlabel('Trials', fontsize=16)  # Increased font size
    plt.ylabel('Item pairs', fontsize=16)  # Increased font size

    # Customize y-axis ticks
    plt.yticks(range(len(average_ordered_pairs)), [str(pair) for pair in average_ordered_pairs], fontsize=14)
    plt.xticks(fontsize=14)

    # Add a legend for triads
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=f'Triad {i} {"(Blocked)" if i <= 3 else "(Interleaved)"}',
                   markerfacecolor='none' if i <= 3 else triad_colors[i],
                   markeredgecolor=triad_colors[i],
                   markersize=10)
        for i in range(1, 7)
    ]
    plt.legend(handles=legend_elements, title='Triad and schedule type',
               loc='upper left',
               title_fontsize=14,  # Increased legend title font size
               fontsize=12)  # Increased legend text font size

    plt.title('Pair Sequence Visualization', fontsize=18)  # Added title with large font size
    plt.tight_layout()
    plt.show()
    plt.savefig('example_sequence.pdf', format='pdf', dpi=300)


# Example usage
pair_of_pairs = create_paired_sampling(seed=42)
complex_pairs, blocked_complex_pairs, interleaved_complex_pairs = generate_complex_pairs_distant(pair_of_pairs)


def save_data_for_r(complex_pairs, pair_of_pairs):
    # Save complex_pairs to a JSON file
    with open('complex_pairs.json', 'w') as f:
        json.dump(complex_pairs, f)

    # Save pair_of_pairs to a JSON file
    with open('pair_of_pairs.json', 'w') as f:
        json.dump(pair_of_pairs, f)

    # Optionally, save to CSV for even more R-friendly format
    import csv

    # Save complex_pairs to CSV
    with open('complex_pairs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Pair', 'TriadNumber'])  # Header
        for pair, triad_number in complex_pairs:
            writer.writerow([pair, triad_number])

    # Save pair_of_pairs to CSV
    with open('pair_of_pairs.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Pair'])  # Header
        for pair in pair_of_pairs:
            writer.writerow([pair])

    print("Files saved: complex_pairs.json, pair_of_pairs.json, complex_pairs.csv, pair_of_pairs.csv")

save_data_for_r(complex_pairs, pair_of_pairs)

# # Count total appearances
# count_total_appearances(complex_pairs, pair_of_pairs)
#
# # Visualize the sequence
visualize_pair_sequence(complex_pairs, pair_of_pairs)
# visualize_pair_sequence(blocked_complex_pairs, pair_of_pairs)
# visualize_pair_sequence(interleaved_complex_pairs, pair_of_pairs)


######################
# model and training #
#####################

# Total number of unique stimuli
NUM_STIMULI = 18

# Extract all unique integers from the pairs
unique_integers = set()
for pair in pair_of_pairs:
    for subpair in pair:
        unique_integers.update(subpair)

# Find the maximum integer to determine the one-hot vector size
max_int = len(unique_integers)
vector_size = 36  # One-hot vectors are indexed from 0 to max_int

# Create one-hot dictionary with 36 entries
one_hot_dict = {num: np.zeros(36) for num in range(36)}
for num in range(36):
    one_hot_dict[num][num] = 1

# Print the resulting one-hot dictionary
for key, value in one_hot_dict.items():
    print(f"{key}: {value}")

# Define the function to get one-hot encodings
def get_one_hot(stimulus):
    return one_hot_dict.get(stimulus, None)


# Function to extract the bottleneck representation for a stimulus
def get_bottleneck_representation_for_stimulus(stimulus, model):
    # Set the model to evaluation mode to disable batch normalization during inference
    model.eval()

    # Use no_grad to avoid tracking gradients
    with torch.no_grad():
        stimulus_input = torch.tensor(get_one_hot(stimulus)).float().unsqueeze(0)  # Add batch dimension
        # Pass through the network to get the bottleneck representation
        bottleneck_rep = model.bottleneck(model.encoder2(model.encoder1(stimulus_input)))  # Forward pass to bottleneck

        # Squeeze the tensor to remove the batch dimension (assuming it has a shape of [1, 56])
        bottleneck_rep = bottleneck_rep.squeeze(0)  # Now bottleneck_rep will have shape [56]

    return bottleneck_rep


def load_model(model_path):
    # Extract parameters from filename
    match = re.search(r'hs_(\d+)_bs_(\d+)', model_path)

    if match:
        hidden_size1 = int(match.group(1))
        bottleneck_size = int(match.group(2))
    else:
        # Raise an error if values cannot be extracted
        raise ValueError(f"Could not extract hidden size and bottleneck size from filename: {model_path}")

    input_size = 36
    hidden_size2 = hidden_size1 // 2

    model = StimulusPredictionNetwork(
        input_size=input_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        bottleneck_size=bottleneck_size
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Example prediction function
def predict_stimulus(input_stimuli):
    # Convert input to one-hot encoding
    input_encoding = torch.tensor(get_one_hot(input_stimuli)).float().unsqueeze(0)  # Add batch dimension

    # Set model to evaluation mode
    model.eval()

    # Predict
    with torch.no_grad():
        prediction = model(input_encoding)

    # Convert back to stimulus number
    predicted_stimulus = torch.argmax(prediction) + 1

    return predicted_stimulus


class StimulusPredictionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, bottleneck_size=56, output_size=None,
                 l1_lambda=1e-2, l2_lambda=1e-2, elastic_beta=0.5):
        super().__init__()

        # Regularization parameters
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.elastic_beta = elastic_beta  # Mixing parameter between L1 and L2

        # If output_size is not provided, use input_size
        output_size = output_size or input_size

        # Encoding layers with Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn_bottleneck = nn.BatchNorm1d(bottleneck_size)

        # Encoding layers
        self.encoder1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            self.bn1,
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            self.bn2,
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size2, bottleneck_size),
            self.bn_bottleneck,
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Decoding layers
        self.decoder2 = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Output layer with sigmoid (for reconstruction)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size1, output_size),
        )

    def forward(self, x):
        # Encoding path
        x = self.encoder1(x)
        x = self.encoder2(x)

        # Bottleneck
        bottleneck_output = self.bottleneck(x)

        # Decoding path
        x = self.decoder2(bottleneck_output)
        x = self.decoder1(x)

        # Final output
        x = self.output_layer(x)

        return x, bottleneck_output  # Return both the output and bottleneck

    def compute_elastic_net_regularization(self):
        """
        Compute Elastic Net regularization (combined L1 and L2)

        Returns:
            torch.Tensor: Regularization loss
        """
        l1_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)

        # Iterate through all parameters
        for param in self.parameters():
            if param.requires_grad:
                # L1 regularization (Lasso)
                l1_reg = l1_reg + torch.norm(param, 1)
                # L2 regularization (Ridge)
                l2_reg = l2_reg + torch.norm(param, 2) ** 2

        # Apply scaling factors for L1 and L2 regularization
        l1_reg_scaled = self.l1_lambda * self.elastic_beta * l1_reg
        l2_reg_scaled = self.l2_lambda * (1 - self.elastic_beta) * l2_reg

        # Combine L1 and L2 regularization terms
        elastic_reg = l1_reg_scaled + l2_reg_scaled

        return elastic_reg

    def pretrain_autoencoder(self, dataloader, criterion, optimizer, num_epochs=100):
        """
        Pre-train the network as an autoencoder with Elastic Net regularization

        Args:
            dataloader (DataLoader): DataLoader containing input data
            criterion (nn.Module): Loss function (BCELoss recommended)
            optimizer (torch.optim.Optimizer): Optimizer for training
            num_epochs (int): Number of pre-training epochs

        Returns:
            float: Final pre-training loss
        """
        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0

            for batch_inputs, _ in dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass - reconstruct input
                reconstructions, bottleneck_output = self(batch_inputs)

                # Compute reconstruction loss
                reconstruction_loss = criterion(reconstructions, batch_inputs)

                # Compute elastic regularization
                elastic_reg_loss = self.compute_elastic_net_regularization()


                # Combine reconstruction loss with elastic regularization
                loss = reconstruction_loss + elastic_reg_loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Compute average epoch loss
            avg_loss = epoch_loss / len(dataloader)

            # Update best loss
            if avg_loss < best_loss:
                best_loss = avg_loss

        return best_loss


def train_stimulus_prediction_network(data, seed, hidden_size1, bottleneck_size, weight_decay, learning_rate, model_num,
                                      data_type, l1_lambda=1e-2, l2_lambda=1e-2, elastic_beta=0.5, num_epochs=125):
    # Hyperparameters
    input_size = vector_size
    hidden_size1 = hidden_size1
    hidden_size2 = hidden_size1 // 2
    bottleneck_size = bottleneck_size
    output_size = vector_size
    learning_rate = learning_rate
    weight_decay = weight_decay

    # Initialize the network with regularization parameters
    torch.manual_seed(seed)
    model = StimulusPredictionNetwork(
        input_size,
        hidden_size1,
        hidden_size2,
        bottleneck_size,
        output_size,
        l1_lambda=l1_lambda,
        l2_lambda=l2_lambda,
        elastic_beta=elastic_beta
    )

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Better weight initialization
    model.apply(init_weights)

    # Prepare data for both pre-training and main training
    inputs = []
    for input_stim in unique_integers:
        input_encoding = get_one_hot(input_stim)
        if input_encoding is None:
            continue
        inputs.append(torch.tensor(input_encoding).float())

    # Convert to tensor
    inputs = torch.stack(inputs)

    # Create DataLoader for pre-training
    pretrain_dataset = TensorDataset(inputs, inputs)  # Same input as target for autoencoder
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=32,
        shuffle=True
    )

    # Pre-training optimizer and loss
    pretrain_criterion = nn.MSELoss()
    pretrain_optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Perform pre-training as an autoencoder
    pretrain_loss = model.pretrain_autoencoder(
        pretrain_dataloader,
        pretrain_criterion,
        pretrain_optimizer
    )

    # Save the model's weights
    model_filename = (
        f"pre_trained_model_{data_type}_"
        f"hs_{hidden_size1}_"
        f"bs_{bottleneck_size}_"
        f"beta_{elastic_beta}_"
        f"{model_num}_weights.pth"
    )
    torch.save(model.state_dict(), os.path.join(model_dir+"_pretrained", model_filename))

    # Metrics tracking DataFrame
    metrics_df = pd.DataFrame(columns=[
        'epoch', 'model_type', 'seed', 'hidden_size1', 'bottleneck_size',
        'stim_A', 'stim_C', 'C_activation', 'cosine_sim',
        'representation_entropy', 'representation_sparsity'
    ])

    # Main training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-5
    )

    # Prepare data for main training
    inputs = []
    targets = []

    for (input_stim, output_stim), _ in data:
        input_encoding = get_one_hot(input_stim)
        target_encoding = get_one_hot(output_stim)

        if target_encoding is None:
            continue

        inputs.append(torch.tensor(input_encoding).float())
        targets.append(torch.tensor(target_encoding).float())

    # Convert to tensors
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    # Create DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    # Tracking variables
    # Tracking metrics every 5 epochs
    def track_metrics():
        pair_metrics = []
        for triad, elements in triad_groups.items():
            # Determine if the triad is blocked or interleaved
            target_pair_condition = 'blocked' if triad in [1, 2, 3] else 'interleaved'

            all_stimuli = [num for sublist in elements for num in sublist]
            counts = Counter(all_stimuli)
            # Keep only numbers that appear once
            stimuli_list = [num for num, count in counts.items() if count == 1]

            stim_A = stimuli_list[0]
            stim_C = stimuli_list[1]

            if (
                    data_type == "blocked_interleaved" or data_type == "interleaved_blocked") and target_pair_condition == "blocked":
                model_type = "pure blocked"
            elif (
                    data_type == "blocked_interleaved" or data_type == "interleaved_blocked") and target_pair_condition == "interleaved":
                model_type = "pure interleaved"
            elif (data_type == "hybrid" or data_type == "hybrid_repeat") and target_pair_condition == "interleaved":
                model_type = "hybrid interleaved"
            elif (data_type == "hybrid" or data_type == "hybrid_repeat") and target_pair_condition == "blocked":
                model_type = "hybrid blocked"

            # Get bottleneck representations for both stimuli
            stim_A_rep = get_bottleneck_representation_for_stimulus(stim_A, model)
            stim_C_rep = get_bottleneck_representation_for_stimulus(stim_C, model)

            # Calculate cosine similarity
            cosine_sim = F.cosine_similarity(stim_A_rep, stim_C_rep, dim=0).item()

            # Get one-hot encoding for stim_A
            stim_A_encoding = get_one_hot(stim_A)

            # Convert input to tensor and add batch dimension
            input_encoding = torch.tensor(stim_A_encoding).float().unsqueeze(0)

            # Predict (assuming this returns the softmaxed output as the first tensor)
            with torch.no_grad():
                prediction = model(input_encoding)
                softmax_output = F.softmax(prediction[0], dim=1)  # Use the first tensor directly

            # Get the activation for stim_C (index is stim_C - 1)
            C_activation = softmax_output[0, stim_C - 1].item()

            # Calculate entropy of bottleneck representation
            bottleneck_rep = stim_A_rep.detach()
            # Normalize the representation to create a probability distribution
            bottleneck_prob = F.softmax(bottleneck_rep, dim=0)
            # Calculate entropy
            entropy = -torch.sum(bottleneck_prob * torch.log2(bottleneck_prob + 1e-10)).item()

            # Calculate sparsity (sum of absolute values of representation)
            sparsity = torch.sum(torch.abs(bottleneck_rep)).item()

            # Add to metrics
            pair_metrics.append({
                'epoch': epoch,
                'model_type': model_type,
                'seed': seed,
                'hidden_size1': hidden_size1,
                'bottleneck_size': bottleneck_size,
                'stim_A': stim_A,
                'stim_C': stim_C,
                'C_activation': C_activation,
                'cosine_sim': cosine_sim,
                'representation_entropy': entropy,
                'representation_sparsity': sparsity
            })

        return pair_metrics

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()

            # Forward pass
            predictions, bottleneck_output = model(batch_inputs)

            # Compute prediction loss
            prediction_loss = criterion(predictions, batch_targets)

            # Compute elastic net regularization loss
            elastic_reg_loss = model.compute_elastic_net_regularization()

            # Combine losses
            total_loss = prediction_loss + elastic_reg_loss

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Update learning rate
        scheduler.step(epoch_loss)

        # Track metrics every 5 epochs
        if epoch % 5 == 0:
            epoch_metrics = track_metrics()
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame(epoch_metrics)
            ], ignore_index=True)

    # Save metrics DataFrame
    metrics_filename = f'metrics_{data_type}_hs_{hidden_size1}_bs_{bottleneck_size}_beta_{elastic_beta}_{model_num}.csv'
    metrics_df.to_csv(os.path.join(model_dir, metrics_filename), index=False)
    print(f"Metrics saved to {metrics_filename}")

    return model


# Train the network
# model = train_stimulus_prediction_network(complex_pairs, seed=42, hidden_size1=256, bottleneck_size=56, weight_decay=0.001, learning_rate=0.001)

# Initialize the dictionary
triad_groups = {}

# Iterate over the tuples and populate the dictionary
for t in complex_pairs:
    key = t[1]  # Third number (second element in the tuple)
    value = tuple(t[0])  # Convert the list to a tuple to ensure it's hashable
    if key not in triad_groups:
        triad_groups[key] = set()  # Initialize an empty set for the key
    triad_groups[key].add(value)  # Add the tuple to the set for the key

    # Convert sets back to lists if needed
triad_groups = {key: tuple([list(v) for v in values]) for key, values in triad_groups.items()}

print(triad_groups)

with open('triad_groups.pkl', 'wb') as f:
    pickle.dump(triad_groups, f)

model_dir = './trained_models_elasticNet_4'  # Replace with the directory where your models are saved

# Train the network multiple times for each scheduler and memory capacities
def train_and_save_multiple_models(num_models=50):
    # Elastic beta sweep values
    elastic_beta_sweep = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    param_grid = [
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 256, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 32, 'bottleneck_size': 18},
        {'learning_rate': 0.001, 'weight_decay': 0.001, 'hidden_size1': 6, 'bottleneck_size': 18},
    ]

    # Loop over each combination of parameters
    counter = 0
    for data in [complex_pairs, complex_pairs, blocked_complex_pairs + interleaved_complex_pairs,
                 interleaved_complex_pairs + blocked_complex_pairs]:
        if counter == 0:
            data_type = "hybrid"
        elif counter == 1:
            data_type = "hybrid_repeat"  # get twice the number of hybrids to match the counterbalanced pure schedules
        elif counter == 2:
            data_type = "blocked_interleaved"
        elif counter == 3:
            data_type = "interleaved_blocked"

        for params in param_grid:
            for elastic_beta in elastic_beta_sweep:
                for model_num in range(num_models):
                    print(f"Training model {model_num + 1}/{num_models} with elastic_beta={elastic_beta}...")

                    # Compute L1 and L2 lambda values
                    l1_lambda = 1e-1
                    l2_lambda = 1e-1

                    model = train_stimulus_prediction_network(
                        data,
                        seed=model_num,
                        hidden_size1=params['hidden_size1'],
                        bottleneck_size=params['bottleneck_size'],
                        weight_decay=params['weight_decay'],
                        learning_rate=params['learning_rate'],
                        model_num=model_num,
                        data_type=data_type,
                        l1_lambda=l1_lambda,
                        l2_lambda=l2_lambda,
                        elastic_beta=elastic_beta
                    )

                    # Save the model's weights
                    model_filename = (
                        f"model_{data_type}_"
                        f"hs_{params['hidden_size1']}_"
                        f"bs_{params['bottleneck_size']}_"
                        f"beta_{elastic_beta}_"
                        f"{model_num}_weights.pth"
                    )
                    torch.save(model.state_dict(), os.path.join(model_dir, model_filename))
                    print(f"Model {model_num + 1} weights saved to {model_filename}")

        counter += 1


# Train and save models
trained_models = train_and_save_multiple_models(50)


# ###############
# # Grid search #
# ###############
#
# def grid_search_stimulus_prediction_network(num_seeds=5):
#     # Hyperparameter ranges
#     hyperparameters = {
#         'learning_rate': [0.01, 0.001, 0.0001],
#         'weight_decay': [0.001, 0.0001, 0.00001],
#         'hidden_size1': [64, 128, 256],
#         'bottleneck_size': [16, 32, 56]
#     }
#
#     # Prepare results storage
#     results = []
#
#     # Generate hyperparameter combinations
#     keys, values = zip(*hyperparameters.items())
#     hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]
#
#     # Prepare data (assuming this is done earlier in your script)
#     inputs = []
#     targets = []
#
#     for (input_stim, output_stim), _ in complex_pairs:
#         input_encoding = get_one_hot(input_stim)
#         target_encoding = get_one_hot(output_stim)
#
#         if target_encoding is None:
#             continue
#
#         inputs.append(torch.tensor(input_encoding).float())
#         targets.append(torch.tensor(target_encoding).float())
#
#     inputs = torch.stack(inputs)
#     targets = torch.stack(targets)
#
#     dataset = TensorDataset(inputs, targets)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#
#     # Grid search with multiple seeds
#     for hp_set in tqdm(hyperparameter_combinations):
#         seed_results = []
#
#         # Run multiple seeds for each configuration
#         for seed in range(num_seeds):
#             # Set seed for reproducibility
#             torch.manual_seed(seed)
#             np.random.seed(seed)
#             random.seed(seed)
#             print(seed)
#
#             # Create model with current hyperparameters
#             model = StimulusPredictionNetwork(
#                 input_size=vector_size,
#                 hidden_size1=hp_set['hidden_size1'],
#                 hidden_size2=hp_set['hidden_size1'] // 2,  # Typically half of first layer
#                 bottleneck_size=hp_set['bottleneck_size'],
#                 output_size=vector_size
#             )
#
#             # Weight initialization
#             def init_weights(m):
#                 if isinstance(m, nn.Linear):
#                     torch.nn.init.xavier_uniform_(m.weight)
#                     m.bias.data.fill_(0.01)
#
#             model.apply(init_weights)
#
#             # Optimizer
#             optimizer = optim.AdamW(
#                 model.parameters(),
#                 lr=hp_set['learning_rate'],
#                 weight_decay=hp_set['weight_decay']
#             )
#
#             # Scheduler
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer,
#                 mode='min',
#                 factor=0.5,
#                 patience=20,
#                 min_lr=1e-5
#             )
#
#             # Training loop
#             best_loss = float('inf')
#             final_loss = float('inf')
#             counter = 0
#
#             for epoch in range(500):  # Fixed max epochs
#                 model.train()
#                 epoch_loss = 0
#
#                 for batch_inputs, batch_targets in dataloader:
#                     outputs = model(batch_inputs)
#                     loss = nn.BCELoss()(outputs, batch_targets)
#                     epoch_loss += loss.item()
#
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#
#                 scheduler.step(epoch_loss)
#
#                 # Early stopping logic
#                 if epoch_loss < best_loss:
#                     best_loss = epoch_loss
#                     counter = 0
#                     final_loss = epoch_loss
#                 else:
#                     counter += 1
#
#                 if counter >= 50:  # Fixed patience
#                     break
#
#             # Store results for this seed
#             seed_results.append({
#                 'seed': seed,
#                 'best_loss': best_loss,
#                 'final_loss': final_loss
#             })
#
#         # Aggregate results across seeds
#         aggregate_result = {
#             **hp_set,
#             'mean_best_loss': np.mean([r['best_loss'] for r in seed_results]),
#             'std_best_loss': np.std([r['best_loss'] for r in seed_results]),
#             'mean_final_loss': np.mean([r['final_loss'] for r in seed_results]),
#             'std_final_loss': np.std([r['final_loss'] for r in seed_results])
#         }
#
#         results.append(aggregate_result)
#         print(f"Completed hyperparameter set: {hp_set}")
#
#     # Convert to DataFrame
#     results_df = pd.DataFrame(results)
#     results_df = results_df.sort_values('mean_best_loss')
#
#     # Save results
#     results_df.to_csv('hyperparameter_search_results.csv', index=False)
#
#     # Print top configurations
#     print("\nTop Hyperparameter Configurations:")
#     print(results_df.head())
#
#     return results_df
#
#
# # Run grid search
# results = grid_search_stimulus_prediction_network(num_seeds=5)

################################
# check results and visualize  #
################################

# Example prediction function
def predict_stimulus(input_stimuli):
    # Convert input to one-hot encoding
    input_encoding = torch.tensor(get_one_hot(input_stimuli)).float().unsqueeze(0)  # Add batch dimension

    # Set model to evaluation mode
    model.eval()

    # Predict
    with torch.no_grad():
        prediction = model(input_encoding)

    # Convert back to stimulus number
    predicted_stimulus = torch.argmax(prediction) + 1

    return predicted_stimulus


integer_mapping = {value: idx + 1 for idx, value in enumerate(sorted(one_hot_dict.keys()))}

# Softmax function
softmax = nn.Softmax(dim=1)  # Change to dim=1 for batch processing

unnested_pairs = [list(item) for pair in pair_of_pairs for item in pair]

# Generate predictions and prepare heatmap data
heatmap_data = []
correct_coords = []

# Convert one_hot_dict keys into a list of possible inputs
possible_inputs = list(one_hot_dict.keys())

# Create a mapping of input to all possible correct outputs from unnested_pairs
input_to_correct_outputs = {}
for pair in unnested_pairs:
    input_val, output_val = pair
    if input_val not in input_to_correct_outputs:
        input_to_correct_outputs[input_val] = []
    input_to_correct_outputs[input_val].append(output_val)

    output_val, input_val = pair # now do for opposite
    if input_val not in input_to_correct_outputs:
        input_to_correct_outputs[input_val] = []
    input_to_correct_outputs[input_val].append(output_val)

# Prepare inputs as a batch
input_batch = []
for input_val in possible_inputs:
    input_encoding = get_one_hot(input_val)
    input_batch.append(input_encoding)

# Convert to tensor with batch dimension
input_tensor = torch.tensor(input_batch).float()

# Get model predictions for entire batch
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = softmax(logits[0]).detach().numpy()

# Iterate over probabilities
for idx, (input_val, probs) in enumerate(zip(possible_inputs, probabilities)):
    # Add probabilities to heatmap data
    heatmap_data.append(probs.tolist())

    # Record correct coordinates for highlighting
    if input_val in input_to_correct_outputs:
        correct_outputs = input_to_correct_outputs[input_val]
        for correct_output in correct_outputs:
            correct_coords.append((idx, integer_mapping[correct_output] - 1))  # Use mapped index

# Convert heatmap data to a NumPy array
heatmap_data = np.array(heatmap_data)

# Visualize the heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data, cmap="viridis", cbar=True, annot=False)

# Highlight correct answers
for row, col in correct_coords:
    ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='red', lw=2))

# Replace x-axis tick labels with keys from one_hot_dict
ax.set_xticklabels([str(key) for key in possible_inputs], rotation=90)
ax.set_yticklabels([str(key) for key in possible_inputs])

# Label axes
plt.xlabel("Predicted Output (as index)")
plt.ylabel("Input (as index)")
plt.title("Model Predictions Heatmap with Correct Outputs Highlighted")
plt.show()

###################
# representations #
###################

with open("./trained_models_elasticNet_4/triad_groups.pkl", 'rb') as file:
    triad_groups = pickle.load(file)

# Iterate over each model file (assuming models are named 'model_*_weights.pth')
model_dir = './trained_models_elasticNet_4'  # Replace with the directory where your models are saved
model_files = [f for f in os.listdir(model_dir) if f.endswith('_weights.pth')]

# Create a list to store all the extracted information
model_info = []

# Loop over each model file and calculate cosine similarity for each model
for triad, elements in triad_groups.items():
    # Determine if the triad is blocked or interleaved
    target_pair_condition = 'blocked' if triad in [1, 2, 3] else 'interleaved'

    all_stimuli = [num for sublist in elements for num in sublist]
    counts = Counter(all_stimuli)
    # Keep only numbers that appear once
    stimuli_list = [num for num, count in counts.items() if count == 1]

    stim_A = stimuli_list[0]
    stim_C = stimuli_list[1]

    # Loop over each model file and calculate cosine similarity for each model
    for model_file in model_files:
        # Extract information from the filename
        match = re.search(r'model_(\w+)_hs_(\d+)_bs_(\d+)_beta_(\d+(?:\.\d+)?)_(\d+)_weights\.pth', model_file)
        # match = re.search(r'model_(\w+)_lr_[\d.e-]+_wd_[\d.e-]+_hs_(\d+)_bs_(\d+)_(\d+)_weights\.pth', model_file)


        if match:
            model_type = match.group(1)

            if model_type == "blocked_interleaved" or model_type == "interleaved_blocked" and target_pair_condition == "blocked":
                model_type = "pure blocked"
            elif model_type == "blocked_interleaved" or model_type == "interleaved_blocked" and target_pair_condition == "interleaved":
                model_type = "pure interleaved"
            elif model_type == "hybrid" or model_type == "hybrid_repeat" and target_pair_condition == "interleaved":
                model_type = "hybrid interleaved"
            elif model_type == "hybrid" or model_type == "hybrid_repeat" and target_pair_condition == "blocked":
                model_type = "hybrid blocked"

            if model_type == "hybrid interleaved" and target_pair_condition == "blocked":
                continue
            elif model_type == "hybrid blocked" and target_pair_condition == "interleaved":
                continue

            hs = int(match.group(2))
            bs = int(match.group(3))
            beta = match.group(4)
            model_number = int(match.group(5))

            model_path = os.path.join(model_dir, model_file)

            # Load the model weights
            model = load_model(model_path)

            # Get bottleneck representations for both stimuli
            stim_A_rep = get_bottleneck_representation_for_stimulus(stim_A, model)
            stim_C_rep = get_bottleneck_representation_for_stimulus(stim_C, model)

            # Calculate cosine similarity
            cosine_sim = F.cosine_similarity(stim_A_rep, stim_C_rep, dim=0).item()

            # Get one-hot encoding for stim_A
            stim_A_encoding = get_one_hot(stim_A)

            # Convert input to tensor and add batch dimension
            input_encoding = torch.tensor(stim_A_encoding).float().unsqueeze(0)

            # Predict (assuming this returns the softmaxed output as the first tensor)
            with torch.no_grad():
                prediction = model(input_encoding)
                softmax_output = prediction[0]  # Use the first tensor directly

            # Get the activation for stim_C (index is stim_C - 1)
            C_activation = softmax_output[0, stim_C - 1].item()

            # Calculate entropy of bottleneck representation
            bottleneck_rep = stim_A_rep.detach()
            # Normalize the representation to create a probability distribution
            bottleneck_prob = F.softmax(bottleneck_rep, dim=0)
            # Calculate entropy
            entropy = -torch.sum(bottleneck_prob * torch.log2(bottleneck_prob + 1e-10)).item()

            # Calculate sparsity (sum of absolute values of representation)
            sparsity = torch.sum(torch.abs(bottleneck_rep)).item()

            # Store all the information
            model_info.append({
                'cosine_sim': cosine_sim,
                'C_activation': C_activation,
                'representation_entropy': entropy,
                'representation_sparsity': sparsity,
                'target_pair_condition': model_type,
                'hs': hs,
                'bs': bs,
                'beta': beta,
                'model_number': model_number,
                'model_type': model_type
            })

# Create a pandas DataFrame
df = pd.DataFrame(model_info)

df['memory'] = df.apply(lambda row: 'medium' if row['hs'] == 32
                        else 'high' if row['hs']==256
                        else 'low' if row['hs']==6
                        else 'unknown', axis=1)

df.to_csv('models_performance_elastic_4.csv', index=False)

########
# Task #
########

triad_ABC_pairs = {}
for triad, elements in triad_groups.items():
    all_stimuli = [num for sublist in elements for num in sublist]
    counts = Counter(all_stimuli)
    # Keep only numbers that appear once (A and C stimuli)
    unique_stims = [num for num, count in counts.items() if count == 1]
    B_stim = [num for num, count in counts.items() if count == 2]
    triad_ABC_pairs[triad] = {'A': unique_stims[0], 'B': B_stim[0], 'C': unique_stims[1]}


def get_model_type(model_file):
    """Determine if model is pure or hybrid from filename"""
    match = re.search(r'model_(\w+)_hs_(\d+)_bs_(\d+)_beta_(\d+(?:\.\d+)?)_(\d+)_weights\.pth', model_file)
    if match:
        base_model_type = match.group(1)
        if base_model_type in ["blocked_interleaved", "interleaved_blocked"]:
            return 'pure'
        elif base_model_type in ["hybrid", "hybrid_repeat"]:
            return 'hybrid'
    return None


def get_schedule_type(model_file):
    """Determine if schedule is blocked or interleaved from filename"""
    match = re.search(r'model_(\w+)_', model_file)
    if match:
        base_model_type = match.group(1)
        if base_model_type.startswith('blocked'):
            return 'blocked'
        elif base_model_type.startswith('interleaved'):
            return 'interleaved'
        elif base_model_type.startswith('hybrid'):
            return 'hybrid'
    return None


def create_test_trials(triad_ABC_pairs):
    """Create all possible test trials while ensuring trials from same triad don't appear consecutively"""
    # Separate triads by condition
    blocked_triads = {k: v for k, v in triad_ABC_pairs.items() if k in [1, 2, 3]}
    interleaved_triads = {k: v for k, v in triad_ABC_pairs.items() if k in [4, 5, 6]}

    def generate_trials(triads):
        trials = []
        for triad_num, items in triads.items():
            # AC trials (C is cue)
            other_triads = {k: v for k, v in triads.items() if k != triad_num}
            for other_triad in other_triads.values():
                trials.append({
                    'triad_num': triad_num,
                    'trial_type': 'AC',
                    'cue': items['C'],
                    'correct': items['A'],
                    'foil': other_triad['A']
                })

            # BC trials (C is cue)
            for other_triad in other_triads.values():
                trials.append({
                    'triad_num': triad_num,
                    'trial_type': 'BC',
                    'cue': items['C'],
                    'correct': items['B'],
                    'foil': other_triad['B']
                })

            # AB trials (B is cue)
            for other_triad in other_triads.values():
                trials.append({
                    'triad_num': triad_num,
                    'trial_type': 'AB',
                    'cue': items['B'],
                    'correct': items['A'],
                    'foil': other_triad['A']
                })
        return trials

    blocked_trials = generate_trials(blocked_triads)
    interleaved_trials = generate_trials(interleaved_triads)

    # Combine and shuffle while ensuring no consecutive trials from same triad
    all_trials = blocked_trials + interleaved_trials
    random.shuffle(all_trials)

    # Reorder if necessary to avoid consecutive same-triad trials
    for i in range(1, len(all_trials)):
        if all_trials[i]['triad_num'] == all_trials[i - 1]['triad_num']:
            # Find next valid trial to swap with
            for j in range(i + 1, len(all_trials)):
                if (all_trials[j]['triad_num'] != all_trials[i - 1]['triad_num'] and
                        (j == len(all_trials) - 1 or all_trials[j]['triad_num'] != all_trials[j + 1]['triad_num'])):
                    all_trials[i], all_trials[j] = all_trials[j], all_trials[i]
                    break

    return all_trials


def evaluate_model(model, trials, model_file):
    """Evaluate model performance on test trials"""
    results = {
        'blocked': {'AC': [], 'BC': [], 'AB': []},
        'interleaved': {'AC': [], 'BC': [], 'AB': []}
    }

    # Determine matrix type from model filename
    matrix_type = get_model_type(model_file)
    if matrix_type is None:
        raise ValueError(f"Could not determine matrix type from filename: {model_file}")

    model.eval()
    with torch.no_grad():
        for trial in trials:
            # Get condition
            condition = 'blocked' if trial['triad_num'] in [1, 2, 3] else 'interleaved'

            # Convert indices to tensors
            cue = trial['cue']
            correct = trial['correct']
            foil = trial['foil']

            # Get embeddings
            cue_emb = get_bottleneck_representation_for_stimulus(cue, model)
            correct_emb = get_bottleneck_representation_for_stimulus(correct, model)
            foil_emb = get_bottleneck_representation_for_stimulus(foil, model)

            # Calculate cosine similarities
            correct_sim = F.cosine_similarity(cue_emb, correct_emb, dim=0).item()
            foil_sim = F.cosine_similarity(cue_emb, foil_emb, dim=0).item()

            # Model chooses option with highest similarity
            correct_choice = correct_sim > foil_sim

            # Store result
            results[condition][trial['trial_type']].append(correct_choice)

    return results


def calculate_accuracy(results):
    """Calculate accuracy overall and by condition/trial type"""
    accuracy = {
        'overall': {},
        'blocked': {},
        'interleaved': {}
    }

    # Calculate accuracy by condition and trial type
    for condition in ['blocked', 'interleaved']:
        for trial_type in ['AC', 'BC', 'AB']:
            trials = results[condition][trial_type]
            if trials:  # Check if there are trials of this type
                accuracy[condition][trial_type] = np.mean(trials)

        # Overall accuracy for condition
        all_trials = sum([results[condition][tt] for tt in ['AC', 'BC', 'AB']], [])
        accuracy[condition]['overall'] = np.mean(all_trials)

    # Overall accuracy across all conditions
    all_trials = sum([sum([results[cond][tt] for tt in ['AC', 'BC', 'AB']], [])
                      for cond in ['blocked', 'interleaved']], [])
    accuracy['overall']['total'] = np.mean(all_trials)

    return accuracy


def run_model_evaluation(model, triad_ABC_pairs, model_file, trials):
    """Main function to run the entire evaluation process"""

    # Evaluate model
    results = evaluate_model(model, trials, model_file)

    # Calculate accuracy
    accuracy = calculate_accuracy(results)

    return accuracy, trials, results


def evaluate_models(model_files, model_dir, triad_ABC_pairs):
    """Evaluate multiple models and compile results into a DataFrame"""
    results_data = []

    # Create test trials
    trials = create_test_trials(triad_ABC_pairs)

    for model_file in tqdm(model_files, desc="Processing models"):
        # Load model
        model_path = os.path.join(model_dir, model_file)
        model = load_model(model_path)
        model_type = get_model_type(model_file)

        # Run evaluation for all trials
        accuracy, trials, results = run_model_evaluation(model, triad_ABC_pairs, model_file, trials)

        match = re.search(r'model_(\w+)_hs_(\d+)_bs_(\d+)_beta_(\d+(?:\.\d+)?)_(\d+)_weights\.pth', model_file)

        hs = int(match.group(2))
        bs = int(match.group(3))
        beta = match.group(4)
        model_number = int(match.group(5))

        # Append data for each trial type and condition
        for trial_type in ['AC', 'BC', 'AB']:
            for condition in ['blocked', 'interleaved']:
                if trial_type in accuracy.get(condition, {}):
                    results_data.append({
                        'model_file': model_file,
                        'trial_type': trial_type,
                        'model_schedule': model_type,
                        'accuracy': accuracy[condition][trial_type],
                        'condition': condition,
                        'hs': hs,
                        'bs': bs,
                        'beta': beta,
                        'model_number': model_number
                    })

    # Create DataFrame
    results_df = pd.DataFrame(results_data)

    return results_df, trials


results_df, trials = evaluate_models(model_files, model_dir, triad_ABC_pairs)

results_df['memory'] = results_df.apply(lambda row: 'medium' if row['hs'] == 32
                        else 'high' if row['hs']==256
                        else 'low' if row['hs']==6
                        else 'unknown', axis=1)
results_df.to_csv('model_evaluation_results_4.csv', index=False)

######
# RSA#
######

# for each model
    # test the model on the task
    # get the triad number for AC trials that are correct

    # get the AC similarity within the correct triads
    # get the similarities within the triads of that schedule
    # set np.nan for similarities between triads across schedules
    # organize the similarities in a 6x6 matrix with the 3 blocked triads first then 3 interleaved

    # for the pretrained
        # get the AC similarity within the correct triads
        # get the similarities within the triads of that schedule
        # set np.nan for similarities between triads across schedules
    # organize the similarities in a 6x6 matrix with the 3 blocked triads first then 3 interleaved

    # get the change in similarity of trained-pretrained matrices

    # was the model pure or hybrid? beta? memory?

    # flatten and save the model's difference matrix to 1x36 and add as columns entries to pandas dataframe.
        # name the columns after the pair that produced the similarity, e.g. A1-C1, A1-C2, A1-C3, etc.
    # add model parameters for memory, beta, hybrid/pure blocked/interleaved as columns to pandas dataframe


def calculate_AC_similarity_matrices(triad_groups, model_files, model_dir, trials):
    # Extract A and C stimuli for each triad
    triad_AC_pairs = {}
    for triad, elements in triad_groups.items():
        all_stimuli = [num for sublist in elements for num in sublist]
        counts = Counter(all_stimuli)
        # Keep only numbers that appear once (A and C stimuli)
        unique_stims = [num for num, count in counts.items() if count == 1]
        triad_AC_pairs[triad] = {'A': unique_stims[0], 'C': unique_stims[1]}

    def create_triad_index_mapping(trials, trial_type):
        mapping = {}
        seen_triads = set()
        current_index = 0

        for trial in trials:
            if trial['trial_type'] == trial_type and trial['triad_num'] not in seen_triads:
                mapping[trial['triad_num']] = current_index
                seen_triads.add(trial['triad_num'])
                current_index += 1
        return mapping

    # Initialize DataFrame to store results
    results_df = pd.DataFrame()

    # Process each model file with progress bar
    for model_file in tqdm(model_files, desc="Processing models"):
        match = re.search(r'model_(\w+)_hs_(\d+)_bs_(\d+)_beta_(\d+(?:\.\d+)?)_(\d+)_weights\.pth', model_file)

        if match:
            model_path = os.path.join(model_dir, model_file)
            model_type = get_model_type(model_file)
            beta = float(match.group(4))
            hs = int(match.group(2))

            # Load trained model
            model = load_model(model_path)

            # Load corresponding pretrained model
            pretrained_file = f"pre_trained_{model_file}"
            pretrained_path = os.path.join('./trained_models_elasticNet_4_pretrained', pretrained_file)
            pretrained_model = load_model(pretrained_path)

            # Evaluate the model to identify correct triads
            evaluation_results = evaluate_model(model, trials, model_file)

            # Create triad to index mapping for AC trials
            triad_index_mapping = create_triad_index_mapping(trials, 'AC')

            # Initialize matrices
            trained_matrix = np.full((6, 6), np.nan)
            pretrained_matrix = np.full((6, 6), np.nan)

            # Loop over each pair of triads
            for triad1 in triad_groups.keys():
                condition1 = 'blocked' if triad1 in [1, 2, 3] else 'interleaved'
                idx1 = triad_index_mapping.get(triad1)

                for triad2 in triad_groups.keys():
                    condition2 = 'blocked' if triad2 in [1, 2, 3] else 'interleaved'
                    idx2 = triad_index_mapping.get(triad2)

                    # Skip between-condition comparisons
                    if condition1 != condition2:
                        continue

                    # Skip if indices not found
                    if idx1 is None or idx2 is None:
                        continue

                    # Check if both triads are correct using the mapped indices
                    if not (evaluation_results[condition1]['AC'][idx1] and
                            evaluation_results[condition2]['AC'][idx2]):
                        continue

                    # Get A and C stimuli representations
                    stim_A = triad_AC_pairs[triad1]['A']
                    stim_C = triad_AC_pairs[triad2]['C']

                    # Get trained model representations and similarity
                    stim_A_rep = get_bottleneck_representation_for_stimulus(stim_A, model)
                    stim_C_rep = get_bottleneck_representation_for_stimulus(stim_C, model)
                    trained_sim = F.cosine_similarity(stim_A_rep, stim_C_rep, dim=0).item()

                    # Get pretrained model representations and similarity
                    stim_A_rep_pre = get_bottleneck_representation_for_stimulus(stim_A, pretrained_model)
                    stim_C_rep_pre = get_bottleneck_representation_for_stimulus(stim_C, pretrained_model)
                    pretrained_sim = F.cosine_similarity(stim_A_rep_pre, stim_C_rep_pre, dim=0).item()

                    # Update both matrices
                    row_idx = triad1 - 1
                    col_idx = triad2 - 1
                    trained_matrix[row_idx, col_idx] = trained_sim
                    pretrained_matrix[row_idx, col_idx] = pretrained_sim

            # Calculate difference matrix
            difference_matrix = trained_matrix - pretrained_matrix

            # Flatten the difference matrix and save to DataFrame
            flattened_diff = difference_matrix.flatten()
            col_names = [f"A{i + 1}-C{j + 1}" for i in range(6) for j in range(6)]
            temp_df = pd.DataFrame([flattened_diff], columns=col_names)

            # Add model parameters to the DataFrame
            temp_df['model_type'] = model_type
            temp_df['beta'] = beta
            temp_df['hs'] = hs

            # Append to results_df
            results_df = pd.concat([results_df, temp_df], ignore_index=True)

    return results_df


RSA_df = calculate_AC_similarity_matrices(triad_groups, model_files, model_dir, trials)

RSA_df['memory'] = RSA_df.apply(lambda row: 'medium' if row['hs'] == 32
    else 'high' if row['hs'] == 256
    else 'low' if row['hs'] == 6
    else 'unknown', axis=1)

RSA_df.to_csv('RSA_elastic_4.csv', index=False)
