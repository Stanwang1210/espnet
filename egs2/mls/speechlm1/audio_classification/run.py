import argparse
import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import logging

from espnet2.speechlm.tokenizer.codec_tokenizer import CodecTokenizer
from espnet2.mt.frontend.embedding import CodecEmbedding

from model import ESC50Model
from dataset import ESC50Dataset, get_dataloader
from utils import set_seed, draw



def get_parser():
    
    parser = argparse.ArgumentParser(description="ESC")
    parser.add_argument(
        "--dumpdir", type=str, default="", help="path to the dump directory"
    )
    parser.add_argument(
        "--exp_dir", type=str, default="", help="path to the exp directory"
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="espnet/amuse_encodec_16k",
        help="HuggingFace model tag for Espnet codec models",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="config file for the frontend",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed"
    )
    
    return parser.parse_args()

def train(model, train_dataloader, criterion, logger, iepoch, accum_grad, log_interval):
    optimizer.zero_grad()
    train_loss, train_acc = [], []
    for ibatch, data in enumerate(train_dataloader):
        utt_id, codec, label = data
        output = model(codec.to("cuda"))
        loss = criterion(output, label)
        loss.backward()
        if ibatch % accum_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        acc = (output.argmax(dim=-1) == label).sum().item() / len(label)
        train_loss.append(loss.item())
        train_acc.append(acc)
        
        if ibatch % log_interval == 0:
            logger.info(
                f"Epoch {iepoch} Batch {ibatch} Loss: {sum(train_loss[-log_interval:]) / log_interval:.3f} Acc: {sum(train_acc[-log_interval:]) / log_interval :.3f}"
            )
        
    logger.info(
        f"Epoch {iepoch} Train Loss: {sum(train_loss) / len(train_loss):.3f} Acc: {sum(train_acc) / len(train_acc):.3f}"
    )
    
    return sum(train_loss) / len(train_loss), sum(train_acc) / len(train_acc)
    
@torch.no_grad()
def valid(model, dev_dataloader, criterion, logger, iepoch):
    dev_loss, dev_acc = [], []
    for ibatch, data in enumerate(dev_dataloader):
        utt_id, codec, label = data
        output = model(codec.to("cuda"))
        loss = criterion(output, label)
        acc = (output.argmax(dim=-1) == label).sum().item() / len(label)
        dev_loss.append(loss.item())
        dev_acc.append(acc)
    
    logger.info(
        f"Epoch {iepoch} Dev Loss: {sum(dev_loss) / len(dev_loss):.3f} Acc: {sum(dev_acc) / len(dev_acc):.3f}"
    )
    
    return sum(dev_loss) / len(dev_loss)  ,sum(dev_acc) / len(dev_acc)

@torch.no_grad()
def test(model_path, test_dataloader, criterion, logger):
    test_acc = []
    test_output = {}
    for ibatch, data in enumerate(test_dataloader):
        utt_id, codec, label = data
        output = model(codec.to("cuda"))
        test_acc += (output.argmax(dim=-1) == label)
        for utt, out, lab in zip(utt_id, output, label):
            test_output[utt] = {"label": lab, "pred": out}
    test_acc = test_acc.sum().item() / len(test_acc)
    logger.info(f"Test Acc: {test_acc:.3f}")
    with open(expdir / 'test_output.json', 'w') as f:
        json.dump(test_output, f)
def main(
    dumpdir: str,
    exp_dir: str,
    model_tag: str,
    config_file: str,
    seed: int,
):
    set_seed(seed)
    dumpdir = Path(dumpdir)
    config_file = Path(config_file)
    
    assert dumpdir.is_dir(), f"{dumpdir} is not a directory"
    assert config_file.exists(), f"{config_file} is not a file"
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    exp_dir = Path(exp_dir) / Path(f"ESC50_{model_tag.replace('/', '_')}_{config_file.stem}")
    os.makedirs(exp_dir, exist_ok=True)
    
        
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(exp_dir / "train.log"),
        ]
    )
    logger = logging.getLogger("train_esc50_main")
    
    logger.info(f"Experiment Directory: {exp_dir}")
    logger.info(f"Data Directory: {dumpdir}")
    logger.info(f"Model Tag: {model_tag}")
    logger.info(f"Config File: {config_file}")
    criterion = nn.CrossEntropyLoss()
    epoch = config.get("epoch", 100)
    accum_grad = config.get("accum_grad", 1)
    log_interval = config.get("log_interval", 10)
    batch_size = config.get("batch_size", 32)
    patience = config.get("patience", 10)
    best_acc = 0.0
    
    train_dataset = ESC50Dataset(dumpdir / "esc50_train")
    dev_dataset = ESC50Dataset(dumpdir / "esc50_dev")
    
    codec_conf = config.get("codec_conf")
    with open(dumpdir / "esc50_train/token_lists/codec_token_list", 'r') as f:
        codec_token_list = f.readlines()

    embedding = CodecEmbedding(input_size=len(codec_token_list), **codec_conf)
    model = ESC50_Model(config, embedding)
    logger.info(f"Model: {model}")
    model.to("cuda")
    
    optim = config.get("optim", "Adam")
    optim_conf = config.get("optim_conf", {"lr": 1e-5})
    optimizer = getattr(torch.optim, optim)(
        model.parameters(), 
        **optim_conf,
    )
    logger.info(f"Optimizer: {optim} Config: {optim_conf}")
    
    if Path(exp_dir / "checkpoint.pth").exists():
        logger.info("Loading Checkpoint")
        checkpoint = torch.load(exp_dir / "checkpoint.pth")
        model.load_state_dict(checkpoint["model"], map_location="cpu")
        model.to("cuda")
        optimizer.load_state_dict(checkpoint["optimizers"])
        start_epoch = checkpoint["epoch"]
        epoch_statistic = checkpoint["statistic"]
        logger.info(f"Resume Training. Start Epoch: {start_epoch}")
    else:
        start_epoch = 0
        epoch_statistic = {"train_loss": [], "train_acc": [], "dev_loss": [], "dev_acc": []}

    
    for iepoch in range(start_epoch, epoch):
        
        logger.info(f"Epoch {iepoch} Start")
        set_seed(iepoch)
        train_dataloader, dev_ddataloader = get_dataloader(train_dataset, dev_dataset, batch_size)
        
        
        train_loss, train_acc = train(model, train_dataloader, criterion, logger, iepoch, accum_grad, log_interval)
        epoch_statistic['train_loss'].append(train_loss)
        epoch_statistic['train_acc'].append(train_acc)
        
            
        dev_loss, dev_acc = valid(model, dev_dataloader, criterion, logger)
        epoch_statistic['dev_loss'].append(dev_loss)
        epoch_statistic['dev_acc'].append(dev_acc)
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(
                model.state_dict().cpu(),
                exp_dir / "best_model.pth",
            )
            patience = config.get("patience", 10)
        else:
            patience -= 1
            if patience == 0:
                logger.info(f"Early Stopping at Epoch {iepoch}")
                break
        
        torch.save(
            {
                "model": model.state_dict().zero_grad().cpu(),
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
        
    test_dataset = ESC50Dataset(
        dumpdir / "esc50_test"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )
    
    logger.info("Testing Start")
    logger.info(f"Loading Best Model from {exp_dir / 'best_model.pth'}")
    test_model = ESC50Model(config, embedding)
    test_model.load_state_dict(torch.load(exp_dir / "best_model.pth"))
    test(test_model, test_dataloader, criterion, logger, exp_dir)
        
        
    
    
    
    
    
    
if '__name__' == '__main__':
    args = get_parser()
    logger.info(args)
    main(**vars(args))