import os
from torch.utils.data import DataLoader, Dataset
import requests
import torch

# Experiment 1: Simple split
# NOTE: We could also use splits from github, not sure what is easiest
experiment_1_train_full = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/tasks_train_simple.txt"
experiment_1_test_full = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/tasks_test_simple.txt"

experiment_1_train_p1 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_train_simple_p1.txt"
experiment_1_train_p2 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_train_simple_p2.txt"
experiment_1_train_p4 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_train_simple_p4.txt"
experiment_1_train_p8 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_train_simple_p8.txt"
experiment_1_train_p16 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_train_simple_p16.txt"
experiment_1_train_p32 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_train_simple_p32.txt"
experiment_1_train_p64 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_train_simple_p64.txt"

experiment_1_test_p1 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_test_simple_p1.txt"
experiment_1_test_p2 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_test_simple_p2.txt"
experiment_1_test_p4 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_test_simple_p4.txt"
experiment_1_test_p8 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_test_simple_p8.txt"
experiment_1_test_p16 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_test_simple_p16.txt"
experiment_1_test_p32 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_test_simple_p32.txt"
experiment_1_test_p64 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/simple_split/size_variations/tasks_test_simple_p64.txt"


# Experiment 2: Train on shorter sequences
experiment_2_train = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/length_split/tasks_train_length.txt"
experiment_2_test = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/length_split/tasks_test_length.txt"
# Experiment 3: Add primitives

experiment_3_tl_train = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_train_addprim_turn_left.txt"
experiment_3_tl_test = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_test_addprim_turn_left.txt"

experiment_3_jump_train_0 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_train_addprim_jump.txt"
experiment_3_jump_test_0 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/tasks_test_addprim_jump.txt"
experiment_3_jump_train_1 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num1_rep1.txt"
experiment_3_jump_test_1 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num1_rep1.txt"
experiment_3_jump_train_2 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num2_rep1.txt"
experiment_3_jump_test_2 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num2_rep1.txt"
experiment_3_jump_train_4 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num4_rep1.txt"
experiment_3_jump_test_4 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num4_rep1.txt"
experiment_3_jump_train_8 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num8_rep1.txt"
experiment_3_jump_test_8 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num8_rep1.txt"
experiment_3_jump_train_16 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num16_rep1.txt"
experiment_3_jump_test_16 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num16_rep1.txt"
experiment_3_jump_train_32 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num32_rep1.txt"
experiment_3_jump_test_32 = "https://raw.githubusercontent.com/brendenlake/SCAN/refs/heads/master/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num32_rep1.txt"






class SCANDataset(Dataset):
    def __init__(self, data_file, max_len):
        self.data = []

        data_dir = os.path.join("data", data_file)
        with open(data_dir, "r") as f:
            src_vocab = set()
            tgt_vocab = set()
            for line in f:
                # The format is "IN: primitive commands OUT: target commands", so the split needs to exclude the "IN:" and "OUT:"
                out_split = line.strip().split("OUT:")
                src = out_split[0].strip().split("IN:")[1].strip()
                tgt = out_split[1].strip()
                for word in src.split():
                    src_vocab.add(word)
                for word in tgt.split():
                    tgt_vocab.add(word)
                self.data.append((src, tgt))

        self.src_vocab = list(src_vocab)
        self.tgt_vocab = list(tgt_vocab)

        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)

        self.src_map = {word: idx for idx, word in enumerate(self.src_vocab)}
        self.tgt_map = {word: idx for idx, word in enumerate(self.tgt_vocab)}
        # Add padding token to the vocabularies
        self.src_map["<PAD>"] = len(self.src_map)
        self.tgt_map["<PAD>"] = len(self.tgt_map)
        # self.src_map["<SOS>"] = len(self.src_map)
        # self.tgt_map["<SOS>"] = len(self.tgt_map)
        # self.src_map["<EOS>"] = len(self.src_map)
        # self.tgt_map["<EOS>"] = len(self.tgt_map)
        self.src_inv_map = {idx: word for idx, word in enumerate(self.src_vocab)}
        self.tgt_inv_map = {idx: word for idx, word in enumerate(self.tgt_vocab)}

        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_words = self.data[idx][0].split()
        tgt_words = self.data[idx][1].split()

        src = [self.src_map[word] for word in src_words]
        tgt = [self.tgt_map[word] for word in tgt_words]
        # src = [self.src_map["<SOS>"]] + src + [self.src_map["<EOS>"]]
        # tgt = [self.tgt_map["<SOS>"]] + tgt + [self.tgt_map["<EOS>"]]
        # Pad the sequences on the right
        src = src + [self.src_map["<PAD>"]] * (self.max_len - len(src))
        tgt = tgt + [self.tgt_map["<PAD>"]] * (self.max_len - len(tgt))

        return torch.tensor(src), torch.tensor(tgt)

    def src_pad_idx(self):
        return self.src_map["<PAD>"]

    def tgt_pad_idx(self):
        return self.tgt_map["<PAD>"]


def make_dataloader(data_file, batch_size, max_len, desired_percentage=1, upscale=False, num_samples=100000):
    """Quickly construct a dataloader for the fetched, for easy import in main script"""
    dataset = SCANDataset(data_file, max_len)

    # Randomly select a subset of the dataset

    subset = int(len(dataset) * desired_percentage)
    print(f"Using {subset} samples out of {len(dataset)}")
    subset_indices = torch.randperm(len(dataset))[:subset]
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

    if upscale:
        sampler = torch.utils.data.RandomSampler(
            subset_dataset, replacement=True, num_samples=num_samples
        )
    else:
        # subset_dataset = dataset  # No need to sample
        sampler = None

    vocabs = [dataset.src_vocab, dataset.tgt_vocab]
    vocab_sizes = [dataset.src_vocab_size, dataset.tgt_vocab_size]
    maps = [dataset.src_map, dataset.tgt_map]
    inv_maps = [dataset.src_inv_map, dataset.tgt_inv_map]
    pad_idxs = [dataset.src_pad_idx(), dataset.tgt_pad_idx()]

    return {
        "dataloader": DataLoader(subset_dataset, sampler=sampler, batch_size=batch_size),
        "vocabs": vocabs,
        "vocab_sizes": vocab_sizes,
        "maps": maps,
        "inv_maps": inv_maps,
        "pad_idxs": pad_idxs,
    }


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
        ("experiment_1_full", experiment_1_train_full, experiment_1_test_full),
        ("experiment_2_full", experiment_2_train, experiment_2_test),
        ("experiment_1_p1", experiment_1_train_p1, experiment_1_test_p1),
        ("experiment_1_p2", experiment_1_train_p2, experiment_1_test_p2),
        ("experiment_1_p4", experiment_1_train_p4, experiment_1_test_p4),
        ("experiment_1_p8", experiment_1_train_p8, experiment_1_test_p8),
        ("experiment_1_p16", experiment_1_train_p16, experiment_1_test_p16),
        ("experiment_1_p32", experiment_1_train_p32, experiment_1_test_p32),
        ("experiment_1_p64", experiment_1_train_p64, experiment_1_test_p64),
        ("experiment_3_tl", experiment_3_tl_train, experiment_3_tl_test),
        ("experiment_3_jump_0", experiment_3_jump_train_0, experiment_3_jump_test_0),
        ("experiment_3_jump_1", experiment_3_jump_train_1, experiment_3_jump_test_1),
        ("experiment_3_jump_2", experiment_3_jump_train_2, experiment_3_jump_test_2),
        ("experiment_3_jump_4", experiment_3_jump_train_4, experiment_3_jump_test_4),
        ("experiment_3_jump_8", experiment_3_jump_train_8, experiment_3_jump_test_8),
        ("experiment_3_jump_16", experiment_3_jump_train_16, experiment_3_jump_test_16),
        ("experiment_3_jump_32", experiment_3_jump_train_32, experiment_3_jump_test_32),
    ]:
        fetch_dataset(train_url, test_url, trial_name)
        print(f"Downloaded {trial_name} dataset")

    print("All datasets downloaded")
