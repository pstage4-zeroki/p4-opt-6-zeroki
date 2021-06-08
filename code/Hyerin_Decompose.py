import wandb
import optuna
import joblib
import copy
import inspect
import argparse
from datetime import datetime
import os
import numpy as np
from typing import Any, Dict, Tuple, Union, List
import pprint

# torch
import torch
import torch.nn as nn

# module/class import
import src
import src.modules
from src.model import Model
from src.utils.common import read_yaml
from src.utils.macs import calc_macs
from train import train

# decompose tensorly
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker

# warning 무시
import warnings 
warnings.filterwarnings('ignore')

"""
module_list : src내 class들을 list로 받는 역할
"""
MODULE_LIST = []
for name, obj in inspect.getmembers(src.modules): 
    if inspect.isclass(obj):
        MODULE_LIST.append(obj)


total_mse_score = 0
len_total_mse_score = 0
def get_mse(src, core,tucker_factors) -> float:
    """ Calc mse for decompose
        src  : pretrain된 conv weight
        tgt : decomposition을 통해 만들어진 conv weight
    """
    global total_mse_score
    global len_total_mse_score

    tgt = tl.tucker_to_tensor((core, tucker_factors))
    if isinstance(src, torch.Tensor):
        total_mse_score += torch.mean((src - tgt)**2)
        len_total_mse_score +=1
    elif isinstance(src, np.ndarray):
        total_mse_score += np.mean((src - tgt)**2)
        len_total_mse_score +=1


class group_decomposition_conv(nn.Module):
    '''
    group 수에 맞춰 소분할 한 후, tucker_decomposition_conv_layer 함수를 거친 x값을 다시 concat 후 return 해줍니다. 
    ex) conv(24,36,kernel_size = 3 , groups = 4)
        -> conv(6,9,kernel_size = 3 , groups = 1)을 decompose하는 과정을총 4번 반복 후 concat
    '''
    def __init__(self, layer : nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.n_groups = layer.groups
        self.in_channel = int(layer.in_channels / self.n_groups)
        self.out_channel = int(layer.out_channels / self.n_groups)

        # tucker_decomposition_conv_layer의 input은 그냥 CONV 껍데기라서 CHANNEL 별로 잘린 X들끼리 같은 CONV를 사용해도 문제 없을 거 같습니다.
        self.conv_module = nn.Conv2d(self.in_channel , self.out_channel , kernel_size = layer.kernel_size , stride = layer.stride , groups = 1, padding = layer.padding, bias = False)
        self.conv_list = []
        for i in range(self.n_groups):
            self.conv_module.weight.data  = layer.weight.data[self.out_channel * i : self.out_channel *(i+1)]
            self.conv_list.append(self.conv_module)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.chunk(self.n_groups , dim = 1)  # 우선 채널 별로 X를 잘라줍니다. 이러면 XS라는 LIST에 담깁니다.
        decompose_x = []
        for value, temp_conv in zip(xs, self.conv_list):
            # XS에 담긴 소분할된 X값 하나하나에 대해서 tucker_decomposition_conv_layer를 통해 나온 nn.Sequential 모듈을 거치게 됩니다.
            decompose_x.append(tucker_decomposition_conv_layer(temp_conv)(value))
        # sequential 거쳐 나온 값들 다시 concat
        out = torch.cat(decompose_x , dim = 1)
        return out


def tucker_decomposition_conv_layer(
      layer: nn.Module,
      normed_rank: List[int] = [0.5, 0.5],
    ) -> nn.Module:
    """Gets a conv layer,
    returns a nn.Sequential object with the Tucker decomposition.
    rank를 받아서 그 rank에 받게 decompositoin한 conv layer들을 sequential 형태로 return 해줌
    """
    if layer.in_channels == 1 or layer.out_channels == 1 :
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
        bias=True if hasattr(layer, "bias") and layer.bias is not None else False,
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = (
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    )
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    get_mse(layer.weight.data, core, [last, first] )
    return nn.Sequential(*new_layers)


def module_decompose(model_layers : nn.Module):
    """
    각 모듈 별 위치에 따른 decompose -> 노가다라서 module이 하나라도 틀리는 경우 안됩니다 ㅜㅜ 
    조건들 고려해서 섬세하게 짜줘야 하는 부분입니다.
    
    새로운 문제점 : 
        depthwise conv에 대해서는 channel이 1이라서 decompose를 해줄 필요가 없다. -> 혼동의 여지가 있어 삭제하지 않고 주석처리 해놨습니다.
        grouped conv일 때 문제가 생긴다. 이때 따로 클래스를 만들어서 처리해주자. -> group_decomposition_conv
    """
    if type(model_layers) == src.modules.conv.Conv :
        if model_layers.conv.groups != 1 and model_layers.conv.in_channels/model_layers.conv.groups != 1 and  model_layers.conv.out_channels/model_layers.conv.groups != 1 :
            model_layers.conv = group_decomposition_conv(model_layers.conv)
        else : 
            model_layers.conv = tucker_decomposition_conv_layer(model_layers.conv)
    
    elif type(model_layers) ==src.modules.dwconv.DWConv:
        if model_layers.conv.groups != 1 and model_layers.conv.in_channels/model_layers.conv.groups != 1 and  model_layers.conv.out_channels/model_layers.conv.groups != 1 :
            model_layers.conv = group_decomposition_conv(model_layers.conv)
        
    elif type(model_layers) == src.modules.shufflev2.ShuffleNetV2 :
        if hasattr(model_layers,'branch1') :
            # model_layers.branch1[0] = tucker_decomposition_conv_layer(model_layers.branch1[0])  # 3x3 depthwise conv라서 생략
            model_layers.branch1[2] = tucker_decomposition_conv_layer(model_layers.branch1[2])
            
        model_layers.branch2[0] = tucker_decomposition_conv_layer(model_layers.branch2[0])
        # model_layers.branch2[3] = tucker_decomposition_conv_layer(model_layers.branch2[3])# 3x3 depthwise conv라서 생략
        model_layers.branch2[5] = tucker_decomposition_conv_layer(model_layers.branch2[5])
    
    elif type(model_layers) == src.modules.mbconv.MBConv:
        # 여기는 아직 생략하기 전입니다. 우선 shufflenet만 depthwise 생략
        if len(model_layers.conv) == 4 :
            # model_layers.conv[0][1] = tucker_decomposition_conv_layer(model_layers.conv[0][1]) # 3x3 depthwise conv라서 생략
            model_layers.conv[1].se[1] = tucker_decomposition_conv_layer(model_layers.conv[1].se[1] , )
            model_layers.conv[1].se[3] = tucker_decomposition_conv_layer(model_layers.conv[1].se[3])
            model_layers.conv[2] = tucker_decomposition_conv_layer(model_layers.conv[2])
        else :
            model_layers.conv[0][1] = tucker_decomposition_conv_layer(model_layers.conv[0][1])
            # model_layers.conv[1][1] = tucker_decomposition_conv_layer(model_layers.conv[1][1])  # 3x3 depthwise conv라서 생략
            model_layers.conv[2].se[1] = tucker_decomposition_conv_layer(model_layers.conv[2].se[1])
            model_layers.conv[2].se[3] = tucker_decomposition_conv_layer(model_layers.conv[2].se[3])
            model_layers.conv[3] = tucker_decomposition_conv_layer(model_layers.conv[3])

    elif type(model_layers) == src.modules.invertedresidualv3.InvertedResidualv3 :
        if len(model_layers.conv) == 6 :
            # model_layers.conv[0] = tucker_decomposition_conv_layer(model_layers.conv[4]) # 3x3 depthwise conv라서 생략
            model_layers.conv[4] = tucker_decomposition_conv_layer(model_layers.conv[4])
        else :
            if type(model_layers.conv[5]) == 'SqueezeExcitation':
                model_layers.conv[0] = tucker_decomposition_conv_layer(model_layers.conv[0])
                # model_layers.conv[3] = tucker_decomposition_conv_layer(model_layers.conv[3])  # 3x3 depthwise conv라서 생략
                model_layers.conv[5].fc1 = tucker_decomposition_conv_layer(model_layers.conv[5].fc1)
                model_layers.conv[5].fc2 = tucker_decomposition_conv_layer(model_layers.conv[5].fc2)
                model_layers.conv[7] = tucker_decomposition_conv_layer(model_layers.conv[7])
            else :
                model_layers.conv[0] = tucker_decomposition_conv_layer(model_layers.conv[0])
                # model_layers.conv[3] = tucker_decomposition_conv_layer(model_layers.conv[3])  # 3x3 depthwise conv라서 생략
                model_layers.conv[7] = tucker_decomposition_conv_layer(model_layers.conv[7])  

    elif type(model_layers) == src.modules.invertedresidualv2.InvertedResidualv2 :
        if len(model_layers.conv) == 3 :
            # model_layers.conv[0][0] = tucker_decomposition_conv_layer(model_layers.conv[0][0]) # 3x3 depthwise conv라서 생략
            model_layers.conv[1] = tucker_decomposition_conv_layer(model_layers.conv[1])
        else :
            model_layers.conv[0][0] = tucker_decomposition_conv_layer(model_layers.conv[0][0]) 
            # model_layers.conv[1][0] = tucker_decomposition_conv_layer(model_layers.conv[1][0]) # 3x3 depthwise conv라서 생략
            model_layers.conv[2] = tucker_decomposition_conv_layer(model_layers.conv[2])


    return model_layers



def decompose(module: nn.Module):
    """model을 받아서 각 module마다 decompose 해줌"""
    model_layers = list(module.children())
    all_new_layers = []
    for i in range(len(model_layers)):
        if type(model_layers[i]) in MODULE_LIST :
            # ex ) Conv , ShuffleNetv2 같은 class 모듈명 
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
        
        tl.set_backend('pytorch')
        # decompose 되기 전 macs 계산
        macs = calc_macs(self.model_instance.model, (3, self.data_config["IMG_SIZE"], self.data_config["IMG_SIZE"]))
        print(f"before decomposition macs: {macs}")  
        
    def __call__(self,trial):
        ############### init for get_mse ######################### 
        global total_mse_score
        global len_total_mse_score  
        total_mse_score = 0
        len_total_mse_score = 0       
        print(self.model_instance)
        ####################### rank setting ############################
        decompose_model =copy.deepcopy(self.model_instance)
        rank1, rank2 = self.search_rank(trial) 
        
        for name, param in decompose_model.model.named_modules():
            if name.split('.')[0] != self.idx_layer:
                self.idx_layer =name.split('.')[0] 
                # serching 이름 구별 용도로 쓰임 # block끼리 같은 rank 쓰는 구조
                rank1, rank2=self.search_rank(trial)        
            if isinstance(param, nn.Conv2d):
                param.register_buffer('rank', torch.Tensor([rank1, rank2])) # rank in, out
            
       ################### decompose model #####################         
        decompose_model.model = decompose(decompose_model.model)
        decompose_model.model.to(device)
        print('---------decompose model check ------------')
        print(decompose_model.model)

        ################### Calculate MACs ########################
        macs = calc_macs(decompose_model.model, (3, self.data_config["IMG_SIZE"], self.data_config["IMG_SIZE"]))
        
        ################### Calculate MSE err ######################## 
        mse_err = total_mse_score / len_total_mse_score

        self.config['mse_err'] = mse_err
        self.config['macs'] = macs

        # for wandb
        wandb.init(project='hyerin', 
                    entity='zeroki',
                    name=f'No_Trial_{trial.number}',
                    group="RankSearch",
                    config=self.config,
                    reinit=True)

        wandb.log({"mse_err": mse_err, "macs": macs}, step=trial.number)

        print('--------------------------------------------------')
        print(f'macs : {macs} , mse_err : {mse_err}')
        print('--------------------------------------------------')
        print('check seaching para initialized : \n', self.config) 

        return mse_err, macs
  
    def search_rank(self,trial):
        para_name1 = f'{self.idx_layer}_layer_rank1'
        para_name2 =f'{self.idx_layer}_layer_rank2'
        rank1 = trial.suggest_float(para_name1, low = 0.03125, high = 0.25, step = 0.03125)
        rank2 = trial.suggest_float(para_name2 , low = 0.03125, high = 0.25, step = 0.03125)        
        self.config[para_name1]=rank1   
        self.config[para_name2]=rank2      
        return rank1, rank2


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="findRank.")
    parser.add_argument(
        "--weight", default="/opt/ml/p4-opt-6-zeroki/code/exp/test/best.pt", type=str, help="model weight path"
    )
    parser.add_argument(
        "--model_config",default="/opt/ml/p4-opt-6-zeroki/code/exp/No_Trial_3_2021-06-07_17-42-12/model.yml", type=str, help="model config path"
    )
    parser.add_argument(
        "--data_config", default="/opt/ml/p4-opt-6-zeroki/code/configs/data/taco_96.yaml", type=str, help="data config used for training."
    )
    parser.add_argument(
        "--run_name", default="rank_decomposition", type=str, help="run name for wandb"
    )
    parser.add_argument(
        "--save_name", default="rank_decomposition", type=str, help="save name"
    )
    args = parser.parse_args()

    data_config = read_yaml(cfg=args.data_config) # 학습시 이미지 사이즈 몇이었는지 확인하는 용도.

    # prepare pretrained weight 
    model_instance = Model(args.model_config, verbose=True)
    # model_instance.model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################################################################################
    study = optuna.create_study(directions=['minimize','minimize'],pruner=optuna.pruners.MedianPruner(
    n_startup_trials=5, n_warmup_steps=5, interval_steps=5))
    study.optimize(func=Objective(model_instance, data_config), n_trials=10)
    joblib.dump(study, '/opt/ml/code/decomposition_optuna.pkl')