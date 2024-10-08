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
    
    plt.plot(train_data, label=f"train_{tag}")
    plt.plot(valid_data, label=f"valid_{tag}")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel(tag)
    plt.title(f"{tag} vs. Epoch")
    plt.savefig(save_path)
    
def get_dataloader(train_dataset, dev_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn,
    )
    
    return train_dataloader, dev_dataloader