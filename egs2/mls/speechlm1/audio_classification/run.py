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
from dataset import ESC50Dataset
from utils import set_seed



def get_parser():
    
    parser = argparse.ArgumentParser(description="ESC")
    parser.add_argument(
        "--data_dir", type=str, default="", help="path to the dump directory"
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

def main(
    data_dir: str,
    exp_dir: str,
    model_tag: str,
    config_file: str,
    seed: int,
):
    set_seed(seed)
    data_dir = Path(data_dir)
    config_file = Path(config_file)
    
    assert data_dir.is_dir(), f"{data_dir} is not a directory"
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
    logger.info(f"Data Directory: {data_dir}")
    logger.info(f"Model Tag: {model_tag}")
    logger.info(f"Config File: {config_file}")
    
    train_dataset = ESC50Dataset(data_dir / "esc50_train")
    dev_dataset = ESC50Dataset(data_dir / "esc50_dev")
    test_dataset = ESC50Dataset(data_dir / "esc50_test")
    
    codec_conf = config.get("codec_conf")
    with open(data_dir / "esc50_train/token_lists/codec_token_list", 'r') as f:
        codec_token_list = f.readlines()
        

    embedding = CodecEmbedding(input_size=len(codec_token_list), **codec_conf)
    model = ESC50_Model(config, embedding)
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
        logger.info(f"Resume Training. Start Epoch: {start_epoch}")
    else:
        start_epoch = 0
    criterion = nn.CrossEntropyLoss()
    
    epoch = config.get("epoch", 100)
    accum_grad = config.get("accum_grad", 1)
    log_interval = config.get("log_interval", 10)
    batch_size = config.get("batch_size", 32)
    patience = config.get("patience", 10)
    best_acc = 0.0
    for iepoch in range(start_epoch, epoch):
        
        logger.info(f"Epoch {iepoch} Start")
        set_seed(iepoch)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )
        
        dev_dataset = DataLoader(
            dev_dataset,
            batch_size,
            shuffle=False,
            collate_fn=dev_dataset.collate_fn,
        )
        
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
            f"Epoch {iepoch} Loss: {sum(train_loss) / len(train_loss):.3f} Acc: {sum(train_acc) / len(train_acc):.3f}"
        )
        
        with torch.no_grad():
            dev_loss, dev_acc = [], []
            for ibatch, data in enumerate(dev_dataloader):
                utt_id, codec, label = data
                output = model(codec.to("cuda"))
                loss = criterion(output, label)
                acc = (output.argmax(dim=-1) == label).sum().item() / len(label)
                dev_loss.append(loss.item())
                dev_acc.append(acc)
            
            logger.info(
                f"Dev Loss: {sum(dev_loss) / len(dev_loss):.3f} Acc: {sum(dev_acc) / len(dev_acc):.3f}"
            )
            
            if sum(dev_acc) / len(dev_acc) > best_acc:
                best_acc = sum(dev_acc) / len(dev_acc)
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
        model_state_dict = model.state_dict().zero_grad().cpu()
        optim_state_dict = optimizer.state_dict()
        torch.save(
            {
                "model": model_state_dict,
                "optimizers": optim_state_dict,
                "epoch": iepoch,
            },
            exp_dir / "checkpoint.pth",
        )
        
    test_dataset = ESC50Dataset(
        data_dir / "esc50_test"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )
    
    with torch.no_grad():
        test_output = {}
        test_loss, test_acc = [], []
        for data in test_dataloader:
            utt_id, codec, label = data
            output = model(codec.to("cuda"))
            
            loss = criterion(output, label)
            acc = (output.argmax(dim=-1) == label).sum().item() / len(label)
            
            for utt, out, lab in zip(utt_id, output, label):
                otest_output[utt] = {"label": lab, "pred": out}
            test_acc.append(acc)
            test_loss.append(loss.item())
        
        
        logger.info(f"Test Loss: {sum(test_loss) / len(test_loss):.3f} Acc: {sum(test_acc) / len(test_acc):.3f}")
        with open('test_output.json', 'w') as f:
            json.dump(test_output, f)
            
    
    
    
    
    
    
if '__name__' == '__main__':
    args = get_parser()
    main(**vars(args))