from scan_dataset import fetch_dataset, make_dataloader
from scan_transformer import Transformer, experiment_hyperparameters, get_device
from scan_train import train_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm

device = get_device()
hyperparameters = experiment_hyperparameters["3"]

# Turn Left
trial_name = "experiment_3_turn_left"
data_dir = f"{trial_name}_train.txt"

# Download datasets
if os.path.exists(data_dir):
    print("Dataset already downloaded.")
else:
    print("Dataset not found. Downloading...")
    experiment3_train = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_train_addprim_turn_left.txt"
    experiment3_test = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_test_addprim_turn_left.txt"

    fetch_dataset(experiment3_train, experiment3_test, trial_name)
    print(f"Downloaded {data_dir}")

train_dataloader_res = make_dataloader(
    data_file=f"{trial_name}_train.txt",
    batch_size=hyperparameters["BATCH_SIZE"],
    max_len=80,
    upscale=True,
)
train_dataloader = train_dataloader_res["dataloader"]
train_pad_idxs = train_dataloader_res["pad_idxs"]

# Initialize the model
model = Transformer(
    src_vocab_size=13 + 3,
    tgt_vocab_size=6 + 3,  # 3 for <PAD>, <SOS>, <EOS>
    src_pad_idx=train_pad_idxs[0],
    tgt_pad_idx=train_pad_idxs[1],
    emb_dim=hyperparameters["EMB_DIM"],
    num_layers=hyperparameters["N_LAYERS"],
    num_heads=hyperparameters["N_HEADS"],
    forward_dim=hyperparameters["FORWARD_DIM"],
    dropout=hyperparameters["DROPOUT"],
).to(device)

# TODO train model

# Save the model
if not os.path.exists("models"):
    os.makedirs("models")

torch.save(model.state_dict(), f"models/{trial_name}.pt")

# Evaluate
model.eval()

test_dataloader_res = make_dataloader(
    data_file=f"{trial_name}_test.txt",
    batch_size=hyperparameters["BATCH_SIZE"],
    max_len=80,
    upscale=True,
)
test_dataloader = test_dataloader_res["dataloader"]

# Evaluation loop
model.eval()
# We are interested in per batch accuracy, as well as overall accuracy both for tokens and sequences
total_token_accuracy = 0
total_sequence_accuracy = 0
total_tokens = 0
total_sequences = 0

with torch.no_grad():
    for i, (src, tgt) in enumerate(tqdm(test_dataloader, desc="Evaluating batches")):
        src = src.to(device)
        tgt = tgt.to(device)

        output = model(src, tgt)
        output = output.argmax(dim=-1)

        if i == 0:
            print("Output shape:", output.shape)
            print("Input example:", src[0])
            print("Output example:", output[0])
            print("Target example:", tgt[0])

        # Calculate token accuracy
        correct_tokens = (output == tgt).sum().item()
        total_tokens += output.numel()
        total_token_accuracy += correct_tokens

        # Calculate sequence accuracy
        correct_sequences = (output == tgt).all(dim=-1).sum().item()
        total_sequences += output.shape[0]
        total_sequence_accuracy += correct_sequences

        # print(
        #     f"Batch {i}, Token Accuracy: {correct_tokens / output.numel():.6f}, Sequence Accuracy: {correct_sequences / output.shape[0]:.6f}"
        # )

print(
    f"Turn Left: Overall Token Accuracy: {total_token_accuracy / total_tokens:.6f}, Overall Sequence Accuracy: {total_sequence_accuracy / total_sequences:.6f}"
)

# Save evaluation results
# with open("experiment-1-results.txt", "a") as f:
#     f.write(
#         f"Split: 2, Token Accuracy: {total_token_accuracy / total_tokens:.6f}, Sequence Accuracy: {total_sequence_accuracy / total_sequences:.6f}, {EPOCHS} epochs \n"
#     )