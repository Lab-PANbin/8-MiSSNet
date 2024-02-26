#!/bin/bash
#SBATCH -o out/MiSS3s.%j.out
#SBATCH -p compute
#SBATCH --qos=normal
#SBATCH -J MiSS3s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=all
#SBATCH --mail-user=your_mail
#SBATCH -w your_node

python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root /your_path/WHDLD \
--batch_size 4 --dataset whdld --name MiSS --task 3s --step 0 --lr 0.01 --epochs 60 --method MiSS --overlap --maxstep 1

python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root /your_path/WHDLD \
--batch_size 4 --dataset whdld --name MiSS --task 3s --step 1 --lr 0.001 --epochs 30 --method MiSS --overlap --maxstep 1 \
--use_csr --weight_csr 0.25 --use_lsd --weight_lsd 0.1 --lsd_sample_size 128

