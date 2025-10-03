#!/bin/bash
#
# Simple SLURM script for submitting multiple serial commands (e.g. parametric studies) 
# using a script wrapper to launch the commands.
#
# To use, change this job script to accommodate 
# running your serial application(s) in your WORKDIR directory (usually the directory of submission).
# Edit the commands file to specify the executions for the launcher (paramrun) to perform.
#
# To run this script, run the follow in the working directory
# sbatch submit_JOB.slurm
#
#-----------------------------------------------------------------------
#                      Get the job Info
#-----------------------------------------------------------------------
# showq -u                  # to show all job under my ID
# squeue -u userID --start  # to get a estimated time
# sstat --jobs=jobID        # stats about active jobs
# scancel jobID             # to cancel a job using its ID
#
#
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#
#         <------------- Setup Parameters ------------>
#                        DO NOT UNCOMMENT
#
#SBATCH -J MNIST                        # Job name
#SBATCH -e jobs/%x_%j_stderr.txt        # Name of stderr error file
#SBATCH -o jobs/%x_%j_stdout.txt        # Name of stdout output file
#SBATCH -p h100                         # Queue (partition) name
#SBATCH --nodes=1                       # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=4             # Total tasks = 2 nodes × 4 GPUs
#SBATCH -t 48:00:00                     # Run time (hh:mm:ss)
#SBATCH -A ASC26                        # Project/Allocation name (req'd if you have more than 1)
#-----------------------------------------------------------------------
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --mail-user=utID@utexas.edu




#-----------------------------------------------------------------------
#                             USING SLURM
# Below, is the script that actually do the job.
# Activate the env, with all the necessery dependencies.
# Pass all the path to input and output files, and scripts.
#-----------------------------------------------------------------------


########################### Run Job ###########################
# Activate the virtual enviroment need to run the job
echo "Activating venv"
source .venv/bin/activate

srun python script/train.py