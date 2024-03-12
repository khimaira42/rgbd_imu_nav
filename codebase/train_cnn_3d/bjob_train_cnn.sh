#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J fusiondataset
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- 
#BSUB -W 18:00 
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "select[sxm2]"
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_fusion_%J.out 
#BSUB -e Output_fusion_%J.err 

# here follow the commands you want to execute with input.in as the input file
source $HOME/miniconda3/bin/activate
conda activate tf-env
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
module load tensorrt/8.6.1.6-cuda-11.X
python $HOME/Documents/thesis/Thesis_code/segmappy/bin/fusiondataset_cnn.py --log fusion_v1 --debug --keep-best