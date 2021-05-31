"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from cmath import nan
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info

import optuna
import joblib
import wandb

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
    optimizer = torch.optim.SGD(
        model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
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
    return test_loss, test_f1, test_acc, macs


class Objective:

    def __init__(self,model_config,data_config):        
        self.model_config = model_config
        self.data_config = data_config        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {}
        
    def __call__(self,trial):
        # model_config 와 data_config 파일을 trial_suggest_form으로 바꿈.
        model_config_suggest, data_config_suggest= self.trial_suggest_form(trial)
        print(data_config_suggest)

        for i,j in data_config_suggest.items(): 
            if i in self.config.keys():
                self.config[i] = j
        print(self.config)         

        log_dir = os.path.join("exp", datetime.now().strftime(f"No_Trial_{trial.number}_%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)

        # Starting WandB run.
        run = wandb.init(project='bohyeon', 
                        entity='zeroki',
                         name=f'No_Trial_{trial.number}',
                         group="sampling",
                         config=self.config,
                         reinit=True)
        try:
            test_loss, test_f1, test_acc,macs = train(
                model_config=model_config_suggest,
                data_config=data_config_suggest,
                log_dir=log_dir,
                fp16=self.data_config["FP16"],
                device=self.device,
            )
        except: 
            #test_loss=nan
            test_f1=0
            #test_acc=0
            macs=nan
            pass

        # WandB logging.
        with run:
            run.log({"test_f1": test_f1, "macs": macs}, step=trial.number)

        return test_f1,macs

    def trial_suggest_form(self,trial):
        # data yaml 파일에서 정보를 불러와 trial_suggest( )로 변경하는 작업
        # model은 구현 못했음. 우선 data 만
        data_config = {}
        for i,j in self.data_config.items(): 
            try:
                if j['suggest']==True:
                    trial_suggest_func=getattr(trial,j['suggest_type'])
                    data_config[i] = trial_suggest_func(i,*j['value'])
                    self.config[i] = data_config[i]
                    print(f'{i} parameter is on trial_suggest.')
            except:
                data_config[i]=j
                pass
        return self.model_config, data_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="configs/model/mobilenetv3.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--data", default="configs/data/taco_automl.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)


    study = optuna.create_study(directions=['maximize','minimize'],pruner=optuna.pruners.MedianPruner(
        n_startup_trials=1, n_warmup_steps=0, interval_steps=1))
    study.optimize(func=Objective(model_config,data_config), n_trials=10)
    joblib.dump(study, '/opt/ml/code/test_optuna.pkl')


