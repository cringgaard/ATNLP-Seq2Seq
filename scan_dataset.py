import os
from torch.utils.data import DataLoader, Dataset
import requests

# Experiment 1: Simple split
experiment_1_train = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/tasks_train_simple.txt"
experiment_1_test = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/tasks_test_simple.txt"

# Experiment 2: Train on shorter sequences
experiment_2_train = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/length_split/tasks_train_length.txt"
experiment_2_test = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/length_split/tasks_test_length.txt"
# Experiment 3: Add primitives

# Experiment 3a: Add primitive jump
experiment3a_train = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_train_addprim_jump.txt"
experiment3a_test = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_test_addprim_jump.txt"

# Experiment 3b: Add primitive turn left
experiment3b_train = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_train_addprim_turn_left.txt"
experiment3b_test = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_test_addprim_turn_left.txt"


class SCANDataset(Dataset):
    def __init__(self, data_file, max_len):
        self.data = []

        data_dir = os.path.join("data", data_file)
        with open(data_dir, "r") as f:
            for line in f:
                # The format is "IN: primitive commands OUT: target commands", so the split needs to exclude the "IN:" and "OUT:"
                out_split = line.strip().split("OUT:")
                src = out_split[0].strip().split("IN:")[1].strip()
                tgt = out_split[1].strip()
                self.data.append((src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: tokenize, pad and return as tensors
        pass


def make_dataloader(data_file, batch_size, shuffle=True):
    """Quickly construct a dataloader for the fetched, for easy import in main script"""
    dataset = SCANDataset(data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fetch_dataset(train_url, test_url, trial_name):
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train_file = os.path.join(data_dir, f"{trial_name}_train.txt")
    test_file = os.path.join(data_dir, f"{trial_name}_test.txt")

    if not os.path.exists(train_file):
        with open(train_file, "wb") as f:
            f.write(requests.get(train_url).content)

    if not os.path.exists(test_file):
        with open(test_file, "wb") as f:
            f.write(requests.get(test_url).content)

    return train_file, test_file


if __name__ == "__main__":
    for trial_name, train_url, test_url in [
        ("experiment_1", experiment_1_train, experiment_1_test),
        ("experiment_2", experiment_2_train, experiment_2_test),
        ("experiment_3a", experiment3a_train, experiment3a_test),
        ("experiment_3b", experiment3b_train, experiment3b_test),
    ]:
        fetch_dataset(train_url, test_url, trial_name)
        print(f"Downloaded {trial_name} dataset")

    print("All datasets downloaded")
