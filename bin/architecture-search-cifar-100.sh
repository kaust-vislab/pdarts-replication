#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --task-per-node=1
#SBATCH --cpu-per-task=2
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=batch
#SBATCH -J architecture-search-cifar-100
#SBATCH -o architecture-search-cifar-100.%J.out
#SBATCH -e architecture-search-cifar-100.%J.err
#SBATCH --mail-user=david.pugh@kaust.edu.sa
#SBATCH --mail-type=ALL

conda activate ./env
python src/pdarts/train_search.py \
  --tmp_data_dir ./data/cifar-100/ \
  --save ./results/logs/cifar-100/ \
  --add_layers 6 \
  --add_layers 12 \
  --dropout_rate 0.1 \
  --dropout_rate 0.4 \
  --dropout_rate 0.7 \
  --cifar100
