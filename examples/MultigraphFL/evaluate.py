import os
import torch
import argparse
import re
from typing import List
from collections import OrderedDict
from examples.MultigraphFL.utils.utils import get_model, get_iterator
from examples.MultigraphFL.task import test


def get_args():
    parser = argparse.ArgumentParser(
        description="Decentralized FL Node"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="femnist",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="benchmark/MultigraphFL/data/femnist/gaia/test/test.pkl",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="benchmark/MultigraphFL/data/femnist/gaia/train/0.pkl",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--round_path",
        type=str,
        default="",
    )
    args = parser.parse_args()
    return args

def average_checkpoints(path: str, device: str="cpu") -> OrderedDict:
    ckpt_files = os.listdir(path)
    ckpt_files = natural_sort(ckpt_files)
    avg_state_dict = OrderedDict()
    num_ckpts = len(ckpt_files)

    for i in range(num_ckpts):
        state_dict = torch.load(os.path.join(path, ckpt_files[i]), map_location=device, weights_only=True)

        if i == 0:
            # Initialize accumulator
            for k, v in state_dict.items():
                avg_state_dict[k] = v.clone().float()
        else:
            for k, v in state_dict.items():
                avg_state_dict[k] += (v.float())

    # Divide by number of checkpoints
    for k in avg_state_dict:
        avg_state_dict[k] /= 11.

    return avg_state_dict

def natural_sort(files: List[str]) -> List[str]:
    def sort_key(name: str):
        nums = re.findall(r'\d+', name)
        if nums:
            return (0, int(nums[0]))   # numeric first
        return (1, float('inf'))       # non-numeric last

    return sorted(files, key=sort_key)

def get_dataloaders(data_name, batch_size_test, test_path, device):
    test_loader = get_iterator(data_name, test_path, device=device, batch_size=batch_size_test, is_test=True)
    train_loader = None
    return train_loader, test_loader

if __name__ == "__main__":
    args = get_args()
    data_name = args.data_name
    test_path = args.test_path
    train_path = args.train_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    optimizer_name = "adam"
    lr_scheduler_name = "sqrt"
    
    train_loader, test_loader = get_dataloaders(data_name, batch_size, test_path, device)
    model = get_model(data_name, device, train_loader if train_loader is not None else test_loader, None, optimizer_name, lr_scheduler_name)
    model.net.to(device)

    round_path = args.round_path
    round_list = os.listdir(round_path)
    round_list = sorted(round_list)
    max_info = ["", 0, 0]
    for i in range(len(round_list)):
        checkpoint_path = os.path.join(round_path, round_list[i])
        state_dict = average_checkpoints(checkpoint_path)
        # state_dict = torch.load(checkpoint_path, weights_only=True)
        model.net.load_state_dict(state_dict)
        model.net.eval()

        loss_round, acc_round = test(model, test_loader, device)
        max_info = [round_list[i], loss_round, acc_round] if acc_round > max_info[2] else max_info
        print(f"Round: {round_list[i]}, loss_round: {loss_round:.4f}, acc_round: {acc_round:.4f}")
    print(max_info)