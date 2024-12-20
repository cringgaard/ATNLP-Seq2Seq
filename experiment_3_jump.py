from scan_dataset import fetch_dataset, make_dataloader
from scan_transformer import Transformer, experiment_hyperparameters, get_device
from scan_train import train_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

EPOCHS = 5
device = get_device()
print("device:", device)
hyperparameters = experiment_hyperparameters["3"]

results = {}

replication = 1

for composed_commands_used in [0,1,2,4,8,16,32]:
    print(f"RUNNING EXPERIMENT WITH COMPOSED COMMANDS USED: {composed_commands_used}")

    trial_name = f"experiment_3_jump_num{composed_commands_used}_rep{replication}"
    data_dir = f"{trial_name}_train.txt"

    # Download datasets
    if os.path.exists(data_dir):
        print("Dataset already downloaded.")
    else:
        print("Dataset not found. Downloading...")
        if composed_commands_used == 0:
            experiment3_train = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_train_addprim_jump.txt" 
            experiment3_test = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_train_addprim_jump.txt"
        else:
            experiment3_train = f"https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num{composed_commands_used}_rep{replication}.txt" 
            experiment3_test = f"https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num{composed_commands_used}_rep{replication}.txt"

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
        print(f"Saved model at {model_path}")
    

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
        f"Jump {composed_commands_used}: Overall Token Accuracy: {total_token_accuracy / total_tokens:.6f}, Overall Sequence Accuracy: {total_sequence_accuracy / total_sequences:.6f}"
    )

    results[f"{composed_commands_used}"] = {
        "token_accuracy": total_token_accuracy / total_tokens,
        "sequence_accuracy": total_sequence_accuracy / total_sequences,
    }

# Extract data for plotting
composed_commands = list(results.keys())  # Treat as categorical labels
token_accuracy = [v["token_accuracy"] * 100 for v in results.values()]
sequence_accuracy = [v["sequence_accuracy"] * 100 for v in results.values()]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Token-Level Accuracy Plot
axes[0].bar(composed_commands, token_accuracy)
axes[0].set_title("Token-Level Accuracy")
axes[0].set_xlabel("Number of Composed Commands Used For Training")
axes[0].set_ylabel("Accuracy on new commands (%)")
axes[0].set_ylim(0, 100)

# Sequence-Level Accuracy Plot
axes[1].bar(composed_commands, sequence_accuracy)
axes[1].set_title("Sequence-Level Accuracy")
axes[1].set_xlabel("Number of Composed Commands Used For Training")
axes[1].set_ylabel("Accuracy on new commands (%)")
axes[1].set_ylim(0, 100)

# Adjust x-axis to treat as categorical labels
axes[0].set_xticks(range(len(composed_commands)))
axes[0].set_xticklabels(composed_commands)

axes[1].set_xticks(range(len(composed_commands)))
axes[1].set_xticklabels(composed_commands)

# Adjust layout and display the plots
plt.tight_layout()

# Save figure
if not os.path.exists("plots"):
    os.makedirs("plots")

plot_name = "experiment3_jump.png"
plot_path = os.path.join("plots", plot_name)

# Check if the file already exists
if os.path.exists(plot_path):
    print(f"Figure '{plot_name}' already exists. Overwrite? (y/n): ", end="")
    user_reply = input().strip().lower()  # Get user input and normalize case
    if user_reply == "y":
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print("Figure overwritten successfully.")
    else:
        print("Figure not saved.")
else:
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print("Figure saved successfully.")
