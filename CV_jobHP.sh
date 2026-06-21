#!/bin/bash

# parameters for slurm
#SBATCH -J Train_CV_model                 # job name, don't use spaces, keep it short

#SBATCH -c 4                         # number of cores, 1

#uncomment if gpu needed
#SBATCH --gres=gpu:1                  # number of gpus 1, some clusters don't have GPUs
#SBATCH --constraint=rtx6000pro|a40|l40 #indicate GPU constraints
#SBATCH --mem=64gb                     # Job memory request

#SBATCH --mail-type=END,FAIL          # email status changes (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --mail-user=t.j.martens@student.utwente.nl   # Where to send mail to

#SBATCH --time=96:00:00                # time limit 1h

#SBATCH --output=slurm_jobs/logs/eeg_preprocess_%j.log      # Standard output and error log

#SBATCH --error=slurm_jobs/logs/eeg_preprocess_%j.err                # if yoou want the errors logged seperately

#SBATCH --partition=bss,main,students#bss, Here 50..is the partition name..can be checked via sinfo
 
  
set -e #exit on error

# Create a directory for this job on the node
ScratchDir="/local/${SLURM_JOBID}"
if [ -d "$ScratchDir" ]; then
   echo "'$ScratchDir' already found !"
else
   echo "'$ScratchDir' not found, creating !"
   mkdir $ScratchDir
fi


# Clean up on the compute node on exit !

cleanup() {
set +e
cd ~
if [ -d "$ScratchDir" ]; then
   echo "'$ScratchDir' found and removing files."
   rm -rf $ScratchDir
   echo "'$ScratchDir' files removed"
else
   echo "Warning: '$ScratchDir' NOT found."
fi
}
trap cleanup EXIT SIGTERM #clean up on exit or time limit reached
# Done.


cd $ScratchDir
 
# Copy input and executable to the node

cp ${SLURM_SUBMIT_DIR}/Models_senn.py $ScratchDir
cp ${SLURM_SUBMIT_DIR}/cross_validation_senn.py $ScratchDir
cp ${SLURM_SUBMIT_DIR}/Read_Data.py $ScratchDir
cp ${SLURM_SUBMIT_DIR}/MyUtils_senn_test.py $ScratchDir


mkdir -p $ScratchDir/Datasets
#cp -r ${SLURM_SUBMIT_DIR}/Datasets/Processed_data $ScratchDir/Datasets
#cp -r ${SLURM_SUBMIT_DIR}/Datasets/. $ScratchDir/Datasets/
#cp -r "/home/s2490390/MasterThesis/BraiNeoCare-main/Datasets/zenodo_eeg" $ScratchDir/Datasets/
#cp -r "/home/s2490390/MasterThesis/BraiNeoCare-main/Datasets/CV_Folds" $ScratchDir/Datasets/
 
cp -r ${SLURM_SUBMIT_DIR}/Datasets/CV_Folds $ScratchDir/Datasets/
# load all modules needed
module purge 
module load miniconda3/25.7
source $(conda info --base)/etc/profile.d/conda.sh #activate conda
echo "conda initialised in shell"

# It's nice to have some information logged for debugging
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Number of nodes used        : "$SLURM_NNODES
echo "Number of MPI ranks         : "$SLURM_NTASKS
echo "Number of threads           : "$SLURM_CPUS_PER_TASK
echo "Number of MPI ranks per node: "$SLURM_TASKS_PER_NODE
echo "Number of threads per core  : "$SLURM_THREADS_PER_CORE
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "
 
caseName=${PWD##*/} # to distinguish several log files
# Run the job -- make sure that it terminates itself before time is up
# Do not submit into the background (i.e. no & at the end of the line).

#Python HP needed for models

MT="$1"
lr="$2"
wd="$3"

robloss="$4"

#run through prepared env
conda run -p /home/s2490390/.conda/envs/STGATbase python $ScratchDir/cross_validation_senn.py --model_type="$MT" --LR="$lr" --WD="$wd" --RobLoss="$robloss"
echo "Job ran succesfully"

# --- Handle History ---
TARGET_HISTORY="${SLURM_SUBMIT_DIR}/History_${SLURM_JOBID}_MT${MT}_LR${lr}_WD${wd}_robloss${robloss}"
mkdir -p "$TARGET_HISTORY"
cp -r "$ScratchDir/History/." "$TARGET_HISTORY/"

# --- Handle Saved_models ---
TARGET_MODELS="${SLURM_SUBMIT_DIR}/Saved_models_${SLURM_JOBID}_MT${MT}_LR${lr}_WD${wd}_robloss${robloss}"
mkdir -p "$TARGET_MODELS"
cp -r "$ScratchDir/Saved_models/." "$TARGET_MODELS/"