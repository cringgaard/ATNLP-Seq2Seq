from scan_dataset import fetch_dataset, make_dataloader
from scan_transformer import Transformer, experiment_hyperparameters, get_device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm

EPOCHS = 5
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
    max_len=50,
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

model_path = f"models/{trial_name}.pt"
if os.path.exists(model_path):
    print("Found existing model")
    model.load_state_dict(torch.load(model_path))
    print(f"loaded model {model_path}")
else:
    print(f"Training model {trial_name}")
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters["LEARNING_RATE"])

    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}: Training batches")):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()

            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1)
            )
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=hyperparameters["GRAD_CLIP"]
            )

            optimizer.step()

            total_loss += loss.item()

    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")

    torch.save(model.state_dict(), model_path)

# Evaluate
model.eval()

test_dataloader_res = make_dataloader(
    data_file=f"{trial_name}_test.txt",
    batch_size=hyperparameters["BATCH_SIZE"],
    max_len=50,
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

output_pad_idx = train_pad_idxs[0]
tgt_pad_idx = train_pad_idxs[1]

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_dataloader, desc="Evaluating batches")):
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        output = model(src, tgt[:, :-1])
        pred_tokens = output.argmax(2)
        # Pred with <SOS> token
        pred_tokens = torch.cat([tgt[:, :1], pred_tokens], dim=-1)

        # Calculate token accuracy in places where tgt and output are not padding tokens
        correct_tokens = ((pred_tokens != output_pad_idx) & (pred_tokens == tgt)).sum().item()
        total_tokens = (tgt != tgt_pad_idx).sum().item()
        total_token_accuracy = correct_tokens / total_tokens

        # Calculate sequence accuracy
        correct_sequences = (pred_tokens == tgt).all(dim=-1).sum().item()
        total_sequences = pred_tokens.shape[0]
        total_sequence_accuracy = correct_sequences / total_sequences

print(
    f"Turn Left: Overall Token Accuracy: {total_token_accuracy / total_tokens:.6f}, Overall Sequence Accuracy: {total_sequence_accuracy / total_sequences:.6f}"
)