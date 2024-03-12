#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J seg_torchrun_new_config
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
### -- specify that the cores must be on the same host -- 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_new_config_%J.out 
#BSUB -e Output_new_config_%J.err 

# here follow the commands you want to execute with input.in as the input file
source $HOME/miniconda3/bin/activate
conda activate forpred
torchrun --standalone --nproc_per_node=1 $HOME/mmdetection/tools/train.py $HOME/mmdetection/configs/haoranDrone/hr_thesis_new_config.py