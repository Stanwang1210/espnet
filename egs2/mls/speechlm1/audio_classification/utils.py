import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def draw(train_data, valid_data, tag, save_path):

    plt.cla()
    plt.clf()
    plt.plot(train_data, label=f"train_{tag}")
    if len(valid_data) > 0:
        plt.plot(valid_data, label=f"valid_{tag}")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel(tag)
    plt.title(f"{tag} vs. Epoch")
    plt.savefig(save_path)
