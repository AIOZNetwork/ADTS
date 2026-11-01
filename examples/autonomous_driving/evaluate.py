import os
import torch
import argparse
import re
from typing import List
from collections import OrderedDict
from examples.autonomous_driving.task import get_iterator, test
from examples.autonomous_driving.models.FADNet import FADNet


def get_args():
    parser = argparse.ArgumentParser(
        description="Decentralized FL Node"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="driving_carla",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="benchmark/FADNet/data/driving_carla/gaia/test/test.npz",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="benchmark/FADNet/data/driving_carla/gaia/train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--round_path",
        type=str,
        default="",
    )
    args = parser.parse_args()
    return args

def average_checkpoints(train_path: str, path: str, device: str="cpu") -> OrderedDict:
    ckpt_files = os.listdir(path)
    ckpt_files = natural_sort(ckpt_files)
    avg_state_dict = OrderedDict()
    num_ckpts = len(ckpt_files)
    # num_samples = get_num_samples(train_path)

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

def get_num_samples(train_path: str) -> List[int]:
    list_files = os.listdir(train_path)
    list_files = natural_sort(list_files)
    list_outs = []
    for i in range(len(list_files)):
        train_path_i = os.path.join(train_path, list_files[i])
        trainloader = get_iterator("driving_carla", train_path_i, "cuda", batch_size=32)
        list_outs.append(len(trainloader.dataset))
    return list_outs

if __name__ == "__main__":
    args = get_args()
    model = FADNet()
    name = args.name
    test_path = args.test_path
    train_path = args.train_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    testloader = get_iterator(name, test_path, device, batch_size=batch_size)

    round_path = args.round_path
    round_list = os.listdir(round_path)
    round_list = sorted(round_list)
    for i in range(len(round_list)):
        checkpoint_path = os.path.join(round_path, round_list[i])
        state_dict = average_checkpoints(train_path, checkpoint_path)
        # state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        loss, rmse = test(model.to(device), testloader, device)
        print(f"Round: {round_list[i]}, Loss: {loss:.4f}, RMSE: {rmse:.4f}")