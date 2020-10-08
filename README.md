# keep_rl (reinforcement learning)
Disclaimer: for entertainment only, do not use for trading decisions.

## Installation

Start keep_dl https://github.com/airocket/keep_dl

install requirements

```sh
pip3 install -r requirements.txt
```
or conda
```sh
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
conda env create -f conda_env\environment.yml
conda activate keep
```
Start
Get models
```sh
python3 main_keep_rl.py
```

Get trade
```sh
python3 main_keep_trade.py
```

Optimization hyperparameter 
```sh
python3 keep_optuna.py
```

TensorBoard 
```sh
tensorboard --logdir tensorboard_keep
```
Web 
```sh
cd web
python3 main_interface.py
```

