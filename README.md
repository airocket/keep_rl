# keep_rl (reinforcement learning)
Disclaimer: for entertainment only, do not use for trading decisions.

## Installation

Start docker Postgres + timescale
```sh
docker-compose up -d --build
```

install requirements

```sh
pip3 install -r requirements.txt
```
Start

```sh
python3 main.py
```

## Usage

Finished models are located in the model directory. Model building studies are in * .ipynb files.
The app tries to predict the KEEP-ETH price using market data and data from the eth.

## Plans for the future

Сollect more historical data and create new models based on them.
