#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --job-name="Gan_tsis" 
#SBATCH --output=salidaGan.out
#SBATCH --gres=gpu:1
#SBATCH --qos=longrunning


# Request a specific node by using: --nodelist=<nodename>
# Request a job for more than two days by using: --qos=longrunning
# Request gpu by using: --gres=gpu:1


echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

#conda activate gan_tsis
#source ./Scripts/activate

echo "Launching experiments"

python main.py --gan_type MyGAN --dataset 4cam --imageDim 256
