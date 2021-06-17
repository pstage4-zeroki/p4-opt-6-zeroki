"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info, seed_everything
from src.scheduler import CosineAnnealingWarmupRestarts
    
def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    #for reproductin 
    seed_everything(data_config['SEED'])
    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.AdamW(
        model_instance.model.parameters(), lr=data_config["INIT_LR"]
    )
    first_cycle_steps = len(train_dl) * data_config["EPOCHS"] /2
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 
        first_cycle_steps=first_cycle_steps, 
        cycle_mult=1.0, 
        max_lr=data_config["INIT_LR"] , 
        min_lr=data_config["INIT_LR"] * 0.1, 
        warmup_steps=int(first_cycle_steps * 0.25), 
        gamma=0.5
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"]+'/train')
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
        fp16 = data_config["FP16"],
        loss_type= "softmax", # softmax, logit_adjustment_loss,F1, Focal, LabelSmoothing
        mix = False # if true : loss = 0.25*crossentropy + loss_type
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="/opt/ml/p4-opt-6-zeroki/code/exp/hyper_heejun_06_10/model.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--data", default="/opt/ml/p4-opt-6-zeroki/code/exp/hyper_heejun_06_10/data.yaml", type=str, help="data config"
    )
    parser.add_argument(
        "--run_name", default="base", type=str, help="run name for wandb"
    )
    parser.add_argument(
        "--save_name", default="base", type=str, help="save name"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("exp", args.save_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # for wandb
    wandb.init(project='bohyeon', entity='zeroki', group ='loss_test_with_model_2',name = args.run_name , save_code = True)
    wandb.run.name = args.run_name
    wandb.run.save()
    wandb.config.update(model_config)
    wandb.config.update(data_config)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
