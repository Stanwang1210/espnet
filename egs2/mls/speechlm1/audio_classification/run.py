import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import logging
import numpy as np
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer
from espnet2.mt.frontend.embedding import CodecEmbedding
from espnet2.torch_utils.model_summary import model_summary

from model import ESC50Model
from dataset import ESC50Dataset, get_dataloader
from utils import set_seed, draw
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    
    parser = argparse.ArgumentParser(description="ESC")
    parser.add_argument(
        "--dumpdir", type=Path, default="", help="path to the dump directory"
    )
    parser.add_argument(
        "--exp_dir", type=Path, default="", help="path to the exp directory"
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="espnet/amuse_encodec_16k",
        help="HuggingFace model tag for Espnet codec models",
    )
    parser.add_argument(
        "--config_file",
        type=Path,
        default="",
        help="config file for the frontend",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "--fold", type=int, default=0, help="fold"
    )
    parser.add_argument(
        "--skip_train", default=False, action="store_true",
    )
    
    return parser.parse_args()

def train(model, train_dataloader, optimizer, criterion, accum_grad, train_count):
    optimizer.zero_grad()
    train_loss, train_acc = [], []
    for count in range(train_count):
        for ibatch, data in enumerate(train_dataloader):
            utt_id, codec, label = data
            output = model(codec.to(device))
            label = label.to(device)
            loss = criterion(output, label)
            loss.backward()
            if ibatch % accum_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            acc = (output.argmax(dim=-1) == label).sum().item() / len(label)
            train_loss.append(loss.item())
            train_acc.append(acc)
        
        # if ibatch % log_interval == 0:
        #     logger.info(
        #         f"Epoch {iepoch} Batch {ibatch} Loss: {sum(train_loss[-log_interval:]) / log_interval:.3f} Acc: {sum(train_acc[-log_interval:]) / log_interval :.3f}"
        #     )
        
    
    return sum(train_loss) / len(train_loss), sum(train_acc) / len(train_acc)
    
@torch.no_grad()
def valid(model, dev_dataloader, criterion):
    dev_loss, dev_acc = [], []
    for ibatch, data in enumerate(dev_dataloader):
        utt_id, codec, label = data
        output = model(codec.to(device))
        label = label.to(device)
        loss = criterion(output, label)
        acc = (output.argmax(dim=-1) == label).sum().item() / len(label)
        dev_loss.append(loss.item())
        dev_acc.append(acc)
    
    
    return sum(dev_loss) / len(dev_loss)  ,sum(dev_acc) / len(dev_acc)

@torch.no_grad()
def test(model, test_dataloader, criterion, logger, expdir):
    test_output = {}
    for ibatch, data in enumerate(test_dataloader):
        utt_id, codec, label = data
        output = model(codec.to(device))
        label = label.to(device)
        for utt, out, lab in zip(utt_id, output.argmax(dim=-1), label):
            test_output[utt] = {"label": lab.item(), "pred": out.item()}
    with open(expdir / 'test_output.json', 'w') as f:
        json.dump(test_output, f)
        
    test_acc = 0
    for k, v in test_output.items():
        if v["label"] == v["pred"]:
            test_acc += 1
    test_acc /= len(test_output)
    logger.info(f"Test Acc: {test_acc:.3f}")
        
        
def main(
    dumpdir: Path,
    exp_dir: Path,
    model_tag: str,
    config_file: Path,
    seed: int,
    fold: int,
    skip_train: bool = False,
):
    set_seed(seed)
    
    assert dumpdir.is_dir(), f"{dumpdir} is not a directory"
    assert config_file.exists(), f"{config_file} is not a file"
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(exp_dir / "images", exist_ok=True)
    
        
    logging.basicConfig(
        filename=str(exp_dir / "train.log"),
        filemode='w',
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    logger = logging.getLogger(f"train_esc50_main_{fold}")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(str(exp_dir / "train.log"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"Data Directory: {dumpdir}")
    logger.info(f"Model Tag: {model_tag}")
    logger.info(f"Config File: {config_file}")
    logger.info(f"Device: {device}")
    criterion = nn.CrossEntropyLoss()
    epoch = config.get("epoch", 100)
    accum_grad = config.get("accum_grad", 1)
    train_count = config.get("train_count", 5)
    batch_size = config.get("batch_size", 32)
    patience = config.get("patience", 10)
    best_acc = 0.0
    
    train_dataset = ESC50Dataset(dumpdir / f"esc50_train_fold_{fold}")
    dev_dataset = ESC50Dataset(dumpdir / f"esc50_dev_fold_{fold}")
    test_dataset = ESC50Dataset(dumpdir / f"esc50_test_fold_{fold}")
    codec_conf = config.get("codec_conf")
    with open(dumpdir / "esc50_train/token_lists/codec_token_list", 'r') as f:
        codec_token_list = f.readlines()

    embedding = CodecEmbedding(input_size=len(codec_token_list), **codec_conf)
    model = ESC50Model(config, embedding)
    logger.info(model_summary(model))
    model.to(device)
    
    optim = config.get("optim", "Adam")
    optim_conf = config.get("optim_conf", {"lr": 1e-5})
    optimizer = getattr(torch.optim, optim)(
        model.parameters(), 
        **optim_conf,
    )
    logger.info(f"Optimizer: {optim} Config: {optim_conf}")
    
    if Path(exp_dir / "checkpoint.pth").exists():
        logger.info("Loading Checkpoint")
        checkpoint = torch.load(exp_dir / "checkpoint.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizers"])
        start_epoch = checkpoint["epoch"]
        epoch_statistic = checkpoint["statistic"]
        logger.info(f"Resume Training. Start Epoch: {start_epoch}")
    else:
        start_epoch = 0
        epoch_statistic = {"train_loss": [], "train_acc": [], "dev_loss": [], "dev_acc": []}

    if not skip_train:
        for iepoch in range(start_epoch, epoch):
            
            logger.info(f"Epoch {iepoch} Start")
            set_seed(iepoch)
            train_dataloader, dev_dataloader, test_dataloader = get_dataloader(train_dataset, dev_dataset, test_dataset, batch_size)
            
            for count in range(train_count):
                train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, accum_grad, train_count)
                logger.info(
                    f"Epoch {iepoch}, Count {count},  Train Loss: {train_loss:.3f} Acc: {train_acc:.3f}"
                )
            epoch_statistic['train_loss'].append(train_loss)
            epoch_statistic['train_acc'].append(train_acc)
            
                
            dev_loss, dev_acc = valid(model, dev_dataloader, criterion)
            logger.info(
                f"Epoch {iepoch}, Dev Loss: {dev_loss:.3f} Acc: {dev_acc:.3f}"
            )
            epoch_statistic['dev_loss'].append(dev_loss)
            epoch_statistic['dev_acc'].append(dev_acc)
            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(
                    model.state_dict(),
                    exp_dir / "best_model.pth",
                )
                patience = config.get("patience", 10)
                logger.info(f"Best Model Saved at Epoch {iepoch}, Best Acc: {best_acc:.3f}")
            else:
                patience -= 1
                if patience == 0:
                    logger.info(f"Early Stopping at Epoch {iepoch}")
                    break
                logger.info(f"Patience: {patience}, Best Acc: {best_acc:.3f}")
            
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizers": optimizer.state_dict(),
                    "epoch": iepoch,
                    "statistic": epoch_statistic,
                },
                exp_dir / "checkpoint.pth",
            )
            
            # Plot curve
            for tag in ["loss", "acc"]:
                draw(
                    epoch_statistic[f"train_{tag}"],
                    epoch_statistic[f"dev_{tag}"],
                    tag,
                    exp_dir / f"images/{tag}.png",
                )
            
    
    logger.info("Testing Start")
    logger.info(f"Loading Best Model from {exp_dir / 'best_model.pth'}")
    test_model = model
    _, _, test_dataloader = get_dataloader(train_dataset, dev_dataset, test_dataset, batch_size)
    test_model.load_state_dict(torch.load(exp_dir / "best_model.pth"), strict=False)
    test(test_model, test_dataloader, criterion, logger, exp_dir)
        
        
    
    
    
    
    
    
if __name__ == '__main__':
    args = get_parser()
    logging.info(args)
    main(**vars(args))