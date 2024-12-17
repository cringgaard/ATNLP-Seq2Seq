from scan_dataset import make_dataloader
from scan_transformer import Transformer, experiment_hyperparameters, get_device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Epochs is not listed in the hyperparameters, so I'll use 15 as a default
EPOCHS = 1

device = get_device()

# I'll use the first experiment hyperparameters for this task
hyperparameters = experiment_hyperparameters["1"]

# Load the dataset
dataloader_res = make_dataloader(
    "experiment_1_train.txt", hyperparameters["BATCH_SIZE"], 80, desired_percentage=0.01
)
train_dataloader = dataloader_res["dataloader"]
train_pad_idxs = dataloader_res["pad_idxs"]

test_dataloader_res = make_dataloader(
    "experiment_1_test.txt", hyperparameters["BATCH_SIZE"], 80, desired_percentage=1
)
test_dataloader = test_dataloader_res["dataloader"]

# Initialize the model
model = Transformer(
    src_vocab_size=13,
    tgt_vocab_size=6,
    src_pad_idx=train_pad_idxs[0],
    tgt_pad_idx=train_pad_idxs[1],
    emb_dim=hyperparameters["EMB_DIM"],
    num_layers=hyperparameters["N_LAYERS"],
    num_heads=hyperparameters["N_HEADS"],
    forward_dim=hyperparameters["FORWARD_DIM"],
    dropout=hyperparameters["DROPOUT"],
).to(device)

# Initialize the optimizer
optimizer = optim.AdamW(model.parameters(), lr=hyperparameters["LEARNING_RATE"])

# Initialize the loss function
criterion = nn.CrossEntropyLoss()

# Training loop

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, (src, tgt) in enumerate(train_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output.view(-1, 6), tgt.view(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=hyperparameters["GRAD_CLIP"]
        )

        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.6f}")


# Evaluation loop
model.eval()
total_loss = 0

with torch.no_grad():
    for i, (src, tgt) in enumerate(test_dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        output = model(src, tgt)
        loss = criterion(output.view(-1, 6), tgt.view(-1))

        total_loss += loss.item()

print(f"Test Loss: {total_loss / len(test_dataloader):.6f}")

# Save the model
if not os.path.exists("models"):
    os.makedirs("models")

torch.save(model.state_dict(), "models/experiment1.pt")
