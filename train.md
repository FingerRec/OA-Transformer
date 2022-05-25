## Install

```
conda env create
pip install decord
pip install ftfy
cd OATrans
mkdir data;
mkdir exps;
```


## Pre-training

### Normal OA-Transformer for retrieval
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 29132 \
train_dist_multi.py \
--config configs/pt/cc3m_webvid/local-region-loss.json # --launcher pytorch
```

### Region-sensitive OA-Transformer for Grounding

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 --master_port 29132 \
train_dist_multi_region_mem.py \
--config configs/pt/cc3m_webvid/local-region-loss.json # --launcher pytorch
```


## Downstream

### zero-shot
```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 \
--master_port 29142 test_region_mem.py --config configs/ft/msrvtt/zsl/normal.json
```

### fine-tuning
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --config configs/ft/msrvtt/fine_tune/normal_1_cl.json
```