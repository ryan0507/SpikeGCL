#!/bin/bash

#SBATCH -J NODE_L
#SBATCH -o out.ARXIV_DASGNN.%j.log
#SBATCH --partition=a6000
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=120:59:59

echo $CUDA_VISIBLE_DEVICES

#HOME dir
cd ~ 
#Working dir
cd SpikeGCL

# python3 -u main.py --neuron PLIF --thr $2 --dataset $1 --T $3 --T_val $3 --aggr $4 --epochs 1000 --sizes 20 10 5
TQDM_DISABLE=1 python3 -u main_dasgnn.py --dataset ogbn-arxiv --T $1 --outs 1 --hids 64 --threshold $2 --no_shuffle --bn --neuron DASGNN --epochs 200 --lr $3

# TQDM_DISABLE=1 python3 -u main_graph_classification.py --neuron LAPLIF --thr 0.25 --dataset PTC_FM --T 5 --T_val 5 --aggr add --epochs 10 --hids 128 128 128 128 --lr 0.001 --root ./datasets/data  --bs 5000 --model SNNGCNN_GC_Degree_Feat --deg_bins -1 --no_db
# TQDM_DISABLE=1 python3 -u main_graph_classification.py --neuron LALIF --thr 0.25 --dataset COLLAB --T 5 --T_val 5 --aggr add --epochs 10 --hids 146 146 146 146 --lr 0.001 --root ./datasets/data  --bs 5000 --model SNNGCNN_GC_Degree --deg_bins 2
# TQDM_DISABLE=1 python3 -u main_graph_classification.py --neuron LALIF --thr 0.25 --dataset MNSIT --T 3 --T_val 3 --aggr add --epochs 10 --hids 146 146 146 146 146 --sizes -1 -1 -1 -1 -1 -1 --lr 0.005 --root ./datasets/data --poisson False --bs 4000
# TQDM_DISABLE=1 python3 -u main_graph_classification.py --neuron LALIF --thr 0.25 --dataset MNIST --T 3 --T_val 3 --aggr add --epochs 10 --hids 146 146 146 146 146 --sizes -1 -1 -1 -1 -1 -1 --lr 0.005 --root ./datasets/data --no_poisson --no_db  --bs 4000

