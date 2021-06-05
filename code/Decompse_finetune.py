import argparse
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
from src.utils.torch_utils import check_runtime, model_info, seed_everything
from src.scheduler import CosineAnnealingWarmupRestarts
import src
import src.modules

import wandb
import optuna
import joblib
import inspect
import argparse
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
from src.utils.torch_utils import check_runtime, model_info, seed_everything
from src.scheduler import CosineAnnealingWarmupRestarts
import src
import src.modules

import wandb
import optuna
import joblib
import inspect

import copy
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from typing import List
from train import train

import warnings 
warnings.filterwarnings('ignore')

"""
module_list : src내 class들을 list로 받는 역할
"""
MODULE_LIST = []
for name, obj in inspect.getmembers(src.modules): 
    if inspect.isclass(obj):
        MODULE_LIST.append(obj)
        
def tucker_decomposition_conv_layer(
      layer: nn.Module,
      normed_rank: List[int] = [0.5, 0.5],
    ) -> nn.Module:
    """Gets a conv layer,
    returns a nn.Sequential object with the Tucker decomposition.
    rank를 받아서 그 rank에 받게 decompositoin한 conv layer들을 sequential 형태로 return 해줌
    """
    if layer.groups != 1:
        return layer

    if hasattr(layer, "rank"):
        normed_rank = getattr(layer, "rank")
    rank = [int(r * layer.weight.shape[i]) for i, r in enumerate(normed_rank)] # channel * normalized rank
    rank = [max(r, 2) for r in rank]

    core, [last, first] = partial_tucker(
        layer.weight.data,
        modes=[0, 1],
        n_iter_max=2000000,
        rank=rank,
        init="svd",
    )

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True,
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = (
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    )
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def module_decompose(model_layers : nn.Module):
    """
    각 모듈 별 위치에 따른 decompose -> 노가다라서 module이 하나라도 틀리는 경우 안됩니다 ㅜㅜ 
    조건들 고려해서 섬세하게 짜줘야 하는 부분입니다.
    """
    if type(model_layers) == src.modules.conv.Conv :
        model_layers.conv = tucker_decomposition_conv_layer(model_layers.conv)
        
    elif type(model_layers) == src.modules.shufflev2.ShuffleNetV2 :
        if hasattr(model_layers,'branch1') :
            model_layers.branch1[0] = tucker_decomposition_conv_layer(model_layers.branch1[0])
            model_layers.branch1[2] = tucker_decomposition_conv_layer(model_layers.branch1[2])
            
        model_layers.branch2[0] = tucker_decomposition_conv_layer(model_layers.branch2[0])
        model_layers.branch2[3] = tucker_decomposition_conv_layer(model_layers.branch2[3])
        model_layers.branch2[5] = tucker_decomposition_conv_layer(model_layers.branch2[5])
    
    elif type(model_layers) == src.modules.mbconv.MBConv:
        if len(model_layers.conv) == 4 :
            model_layers.conv[0][1] = tucker_decomposition_conv_layer(model_layers.conv[0][1])
            model_layers.conv[1].se[1] = tucker_decomposition_conv_layer(model_layers.conv[1].se[1])
            model_layers.conv[1].se[3] = tucker_decomposition_conv_layer(model_layers.conv[1].se[3])
            model_layers.conv[2] = tucker_decomposition_conv_layer(model_layers.conv[2])
        else :
            model_layers.conv[0][1] = tucker_decomposition_conv_layer(model_layers.conv[0][1])
            model_layers.conv[1][1] = tucker_decomposition_conv_layer(model_layers.conv[1][1])
            model_layers.conv[2].se[1] = tucker_decomposition_conv_layer(model_layers.conv[2].se[1])
            model_layers.conv[2].se[3] = tucker_decomposition_conv_layer(model_layers.conv[2].se[3])
            model_layers.conv[3] = tucker_decomposition_conv_layer(model_layers.conv[3])        
    return model_layers



def decompose(module: nn.Module):
    """model을 받아서 각 module마다 decompose 해줌"""
    model_layers = list(module.children())
    all_new_layers = []
    for i in range(len(model_layers)):
        if type(model_layers[i]) in MODULE_LIST :
            # MODULE_LIST 내 존재하는 MODULE일 경우
            all_new_layers.append(module_decompose(model_layers[i]))
        
        elif type(model_layers[i]) == nn.Sequential:
            # Sequential일 경우 재귀 형태로 return값을 받아줘서 temp리스트에 저장 후 그걸 최종적으로 all_new_layer라는 전체 list에 append해줌
            temp = []
            for j in range(len(model_layers[i])):
                temp.append(module_decompose(model_layers[i][j]))
            all_new_layers.append(nn.Sequential(*temp))
        
        else :
            # Linear,maxpooling등의 클래스인 경우 그냥 그대로 append
            all_new_layers.append(model_layers[i])
    
    return nn.Sequential(*all_new_layers)



class Objective:
    def __init__(self, model_instance, data_config):     
        
        self.model_instance = model_instance # 학습된 원 모델
        self.idx_layer = 0
        self.data_config = data_config
        self.config = {}
        
        macs = calc_macs(self.model_instance.model, (3, self.data_config["IMG_SIZE"], self.data_config["IMG_SIZE"]))
        print(f"before decomposition macs: {macs}")  
        tl.set_backend('pytorch')
     

    def __call__(self,trial):       
        
        ####################### rank 설정 ############################
        decompose_model = self.model_instance
        rank1, rank2 = self.search_rank(trial) 
 
        for name, param in decompose_model.model.named_modules():
            if name.split('.')[0] != self.idx_layer:
                self.idx_layer =name.split('.')[0] 
                # serching 이름 구별 용도로 쓰임 # block끼리 같은 rank 쓰는 구조
                rank1, rank2=self.search_rank(trial)        
            if isinstance(param, nn.Conv2d):   
                param.register_buffer('rank', torch.Tensor([rank1, rank2])) # rank in, out
                
       ################### decompose model 불러오는 함수 #####################         
     
        decompose_model.model = decompose(decompose_model.model)
        decompose_model.model.to(device)
        print('---------decompose model check ------------')
        print(decompose_model.model)

        ################### Calculate MACs ############
        macs = calc_macs(decompose_model.model, (3, self.data_config["IMG_SIZE"], self.data_config["IMG_SIZE"]))
        print(f"after decomposition macs: {macs}")        

        log_dir = os.path.join("exp_decomp", datetime.now().strftime(f"No_Trial_{trial.number}_%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        
        # Starting WandB run. # 우선 꺼둠.
        '''
        run = wandb.init(project='bohyeon', 
                        entity='zeroki',
                         name=f'No_Trial_{trial.number}',
                         group="RankSearch",
                         config=self.config,
                         reinit=True)
        '''
    
        # 추가 구현되어야 하는거 : mse 계산 (각 conv2d decomposition 될 때 한번에 계산하는게 편하지 않을까 생각중
        # 구현을 한다면 tucker_decomposition_conv_layer 함수안에서 매번 계산하는게 편할 거 같음. 


        # WandB logging.
        '''
        with run:
            run.log({"mse_err": mse_err, "macs": macs}, step=trial.number)
        '''
        self.config = {}
        print('check seaching para initialized : \n', self.config) 
        return mse_err, macs    

    def search_rank(self,trial):
        para_name1 = f'{self.idx_layer}_layer_rank1'
        para_name2 =f'{self.idx_layer}_layer_rank2'
        rank1 = trial.suggest_float(para_name1, low = 0.03125, high =1.0, step = 0.03125)
        rank2 = trial.suggest_float(para_name2 , low = 0.03125, high =1.0, step = 0.03125)        
        self.config[para_name1]=rank1   
        self.config[para_name2]=rank2      
        return rank1, rank2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="findRank.")
    parser.add_argument(
        "--weight", default="/opt/ml/p4-opt-6-zeroki/code/exp/base_2021-06-03_03-17-29/best.pt", type=str, help="model weight path"
    )
    parser.add_argument(
        "--model_config",default="/opt/ml/p4-opt-6-zeroki/code/exp/base_2021-06-03_03-17-29/model.yml", type=str, help="model config path"
    )
    parser.add_argument(
        "--data_config", default="/opt/ml/p4-opt-6-zeroki/code/exp/base_2021-06-03_03-17-29/data.yml", type=str, help="dataconfig used for training."
    )
    parser.add_argument(
        "--run_name", default="rank_decomposition", type=str, help="run name for wandb"
    )
    parser.add_argument(
        "--save_name", default="rank_decomposition", type=str, help="save name"
    )

    args = parser.parse_args()
    data_config = read_yaml(cfg=args.data_config) # 학습시 이미지 사이즈 몇이었는지 확인하는 용도.

     # 학습된 모델(weight같이) 불러 옴 
    model_instance = Model(args.model_config, verbose=True)
    model_instance.model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################################################
    study = optuna.create_study(directions=['minimize','minimize'],pruner=optuna.pruners.MedianPruner(
    n_startup_trials=5, n_warmup_steps=5, interval_steps=5))
    study.optimize(func=Objective(model_instance, data_config), n_trials=2)
    joblib.dump(study, '/opt/ml/code/decomposition_optuna.pkl')
