# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
./download.sh
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent Training

如果想要重現最佳modle的話，請跑

```shell
python train_intent.py --data_dir <directory of data>
```

如果想要重現調參的過程的話，請跑

```shell
python train_intent.py --FindParam
```

## Slot Tagging Training

如果想要重現最佳modle的話，請跑

```shell
python train_slot.py --dropout=0.2 --data_dir <directory of data>
```

如果想要重現調參的過程的話，請跑

```shell
python train_slot.py --FindParam
```

## Results

| Model                 | Public Score | Private Score | Rank   |
| --------------------- | ------------ | ------------- | ------ |
| Intent Classification | 0.92133      | 0.92044       | 70/196 |
| Slot Tagging          | 0.78498      | 0.78188       | 77/190 |

