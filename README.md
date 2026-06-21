# Cluster-GT Classification Workflow

This directory keeps the runnable classification workflow, datasets, and the
best IMDB configuration files.The complete source code will be made publicly available upon acceptance of MCGRL.

## Contents

```text
.
├── main.py
├── eval.py
├── data.py
├── dataset_utils.py
├── dataset_transform.py
├── pe.py
├── my_utils.py
├── model/
├── datasets/
│   ├── IMDB-BINARY/
│   ├── IMDB-MULTI/
│   ├── COLLAB/
│   ├── MUTAG/
│   ├── PROTEINS/
│   ├── DD/
│   └── NCI1/
├── config/
│   ├── IMDB-BINARY.json
│   └── IMDB-MULTI.json
├── output/
├── logs/
├── requirements.txt
└── README.md
```

## Environment

The original environment used:

```text
python==3.8.18 or 3.9
pytorch==1.8.1
torch_geometric==1.6.3
torch_sparse==0.6.12
torch_scatter==2.0.8
torch_cluster==1.5.9
torch_spline_conv==1.2.1
```

Additional Python packages are listed in `requirements.txt`.

## Run IMDB-BINARY Example

`main.py` defaults to `IMDB-BINARY`.

Use the best parameters from `results_IMDB-BINARY.csv`:

```bash
cd /data/zzh/paper/classification/code
python main.py --config config/IMDB-BINARY.json --gpu 0
```

Equivalent explicit command:

```bash
python main.py --data IMDB-BINARY --gpu 0 \
  --num-hidden 64 \
  --weight-decay 0.0001741329771582 \
  --lr 0.0001038768039137 \
  --batch-size 64
```

## Run IMDB-MULTI Best Config

```bash
python main.py --config config/IMDB-MULTI.json --gpu 0
```

## Run Other Datasets

Available datasets:

```text
IMDB-BINARY IMDB-MULTI COLLAB MUTAG PROTEINS DD NCI1
```

Example:

```bash
python main.py --data DD --gpu 0
```

Results are appended to:

```text
output/result.csv
```
