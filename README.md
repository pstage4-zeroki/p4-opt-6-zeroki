# 부스트캠프 AI TECH <Pstage 4 경량화>

## 🙋‍♀️ 팀 소개 🙋‍♂️

- 모델 최적화 6조 (영기일기이기삼기사기 조)
- 조원 : 박성훈, 엄희준, 오혜린, 이보현, 장보윤

|                                                                                      박성훈                                                                                      |                                                             엄희준                                                             |                                                          오혜린                                                           |                                                            이보현                                                            |                                                            장보윤                                                             |                                                            
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: | 
| <a href='https://github.com/seong0905'><img src='https://avatars.githubusercontent.com/u/70629496?v=4' width='200px'/></a> | <a href='https://github.com/eomheejun'><img src='https://avatars.githubusercontent.com/u/50470448?v=4' width='200px'/></a> | <a href='https://github.com/Hyerin-oh'><img src='https://avatars.githubusercontent.com/u/68813518?s=400&u=e5300247dc2b04f5cf57265a6f2e1cc0987e6d08&v=4' width='200px'/></a> | <a href='https://github.com/bonniehyeon'><img src='https://avatars.githubusercontent.com/u/50580028?v=4' width='200px'/></a> | <a href='https://github.com/dataminegames'><img src='https://avatars.githubusercontent.com/u/45453533?v=4' width='200px'/></a> | 


#### 프로젝트 일정 
- 21.05.24 ~ 

![image](https://user-images.githubusercontent.com/50580028/119325878-52906380-bcbc-11eb-83ca-ca20efd3f06d.png)

# Run
## 1. train single file
```
python train.py
```
## 2. AutoML for Architecture Searching(NAS)
```
python tune_architecture.py
```
## 3. AutoML for Hyper Parameter Searching
```
python tune_hyper.py
```
## 4. Decompose Architecture
```
python train.py
```
## 5. inference(submission.csv)
```
python inference.py --model_config [model config file path] --weight [weight file path] --img_root /opt/ml/data/test --data_config [data config file path]
```
