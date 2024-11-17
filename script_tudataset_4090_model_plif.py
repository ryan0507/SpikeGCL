import os
import time
import subprocess
from itertools import product

GPU_TYPE = 'a6000'

def run_or_wait(command):
    not_done = True
    while not_done:
        queue_stat = subprocess.check_output(['squeue'],encoding='utf-8')
        queue_lst = queue_stat.split('\n')
        cnt = sum(GPU_TYPE in l for l in queue_lst)
        if 31 - cnt < 1:
            left = 31 - cnt
            print(f"manner maketh man. The queue has {left} sits left")
            time.sleep(61)
            continue
        try:
            output = subprocess.check_output(command,encoding='utf-8')#command)
            print(output)
            print(command,"is good")
            not_done = False
        except BaseException as e:
            print(e)
            print(command,"is wating to be assigned to queue...")
            time.sleep(60)

# ID_LIST = ["01","02","03"]
# NUM_LAYER = "5"

# DATASET = ['MNIST', 'CIFAR10']
# DATASET = ['COLLAB']
# DATASET = ['MUTAG']

# DATASET = ['MUTAG']
# DATASET = ['MUTAG', 'ENZYMES', 'IMDB-BINARY', 'NCI1', 'PROTEINS']
# DATASET = ['MUTAG', 'ENZYMES', 'NCI1']
DATASET = ['COLLAB']

# DATASET = ['NCI1', 'PROTEINS', 'IMDB-BINARY-DEG']
# DATASET = ['ENZYMES', 'PROTEINS']
# DATASET = ["REDDIT-BINARY"]
# DATASET = ["IMDB-BINARY-DEG"]

# DATASET = ['MUTAG']

# DATASET = ['IMDB-BINARY']
# NEURON = ["PLIF"]
# NEURON = ["LIF_tr", "LPLIF"]
# NEURON = ["LPLIF"]
# NEURON = ['LIF_same']
NEURON = ["LAPLIF"]
# NEURON = ["PLIF", 'LIF_same',"ALIF"]

# NEURON = ["TWOLAPLIF"]
THR = [2.5]
# THR = [2.5]
# THR = [0.5, 1.0, 1.5, 2.0, 3.0,3.5,4.0]
# THR = [10.0, 6.5, 6.0, 6.0, 5.5, 5.0, 4.5, 4.0,3.5,1.0, 0.5]
# THR = [0.5,1.5,5.0,7.0,10.0]


# THR = [30.0]
# T = [1,2,3,4,5,6,7]
# T = [25, 50]
T = [5]
AGGR = ['add']
# LR = [0.001, 0.005, 0.05, 0.1, 0.5, 0.01]
LR = [0.01]
# MODEL = ["SNNGIN_GC_Degree_Feat"]
# MODEL = ["SNNGCNN_GC_Degree_Feat", "SGAT_Degree_Feat","SNNGIN_GC_Degree_Feat"]
# MODEL = ["SNNGCNN_GC_Degree_Feat","SNNGIN_GC_Degree_Feat"]
# MODEL = ["SNNGCNN_GC_Degree_Feat"]

# MODEL = ["SNNGIN_GC_Degree_Feat_fixed","SNNGCNN_GC_Degree_Feat"]

MODEL = ["SGAT_Degree_Feat"]

# MODEL = ["SNNGIN_GC_Degree_Feat_fixed"]

BINS = [-1]
# SEED = [7777,7777,7777,7777,7777,7777,7777, 7777, 7777,7777, 7777,7777,7777,7777,
        # 7777,7777,7777,7777,7777,7777,7777,7777,7777,7777,7777,7777,7777,7777]
SEED=[7777]
# GAMMA = [0.30, 0.35, 0.40, 0.45, 0.50]
GAMMA = [0.20]

# MODEL = ["SNNGCNN_GC"]
# BINS = [-1]
# BINS = [1,2,3]

# BINS = [2,5,10]
# BINS = [2,3,4,5,6,7,8,9,10]
# SURROGATE = ['triangle', "sigmoid", "arctan", 'super']
# LR = [0.1, 0.05, 0.01, 0.005]
# HIDDEN_DIM = ['64','128']
# BINS=[1,2]
# BINS = [2,3,4,5,6,7,8,9,10]
# SURROGATE = ['triangle', "sigmoid", "arctan", 'super']
# LR = [0.1, 0.05, 0.01, 0.005]
# HIDDEN_DIM = ['64','128']

T = [8]
thr = [0.005,0.05,0.1,0.5]
n = ["ALIF"]
LR = [0.001, 0.005, 0.01, 0.05, 0.1]
    
for t, threshold, neuron, lr in product(T, thr,n, LR):
    # command = ["sbatch","--exclude","aisys-cluster03", "go_one_train_t.sh",
    command = ["sbatch","script_tudataset_4090_model_plif.sh",
               str(t), str(threshold), str(neuron), str(lr)]
    run_or_wait(command)
    time.sleep(3)