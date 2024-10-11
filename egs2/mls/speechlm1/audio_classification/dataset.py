import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import kaldiio


class ESC50Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.wavscp = self.data_dir / "index_files/wav.scp"
        self.text = self.data_dir / "index_files/text"
        
        self.codecs = kaldiio.load_scp(str(self.wavscp))
        self.labels = {}
        with open(self.text, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, txt = line.strip().split(maxsplit=1)
                txt = txt.split(":")[-1]
                self.labels[utt_id] = int(txt)
                
        self.utt_ids = list(self.codecs.keys())

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        codec = self.codecs[utt_id]
        label = self.labels[utt_id]
        return utt_id, codec, label

    def collate_fn(self, batch):
        utt_id, codec, label = zip(*batch)
        codec = torch.stack([torch.from_numpy(c) for c in codec])
        label = torch.tensor(list(label))
        return utt_id, codec, label
        
def get_dataloader(train_dataset, dev_dataset, test_dataset, batch_size):
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
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )
    
    return train_dataloader, dev_dataloader, test_dataloader