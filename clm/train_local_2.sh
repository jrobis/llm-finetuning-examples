#!/bin/bash

# Define the file to store the input values
INPUT_FILE="input.txt"

# Read the input values from the file, or set default values if the file doesn't exist
if [ -f $INPUT_FILE ]; then
  read seq_length examples epochs lr algo batches local_model model_input model_output numgpu < $INPUT_FILE
else
  seq_length=256
  examples=512
  epochs=100
  lr=0.0005
  algo="AdamW"
  batches=12
  local_model="Yes"
  model_input="EleutherAI/gpt-j-6B"
  model_output="model_2.7B"
  numgpu=0
fi

# Prompt the user for the input values, preloading the values from the file
read -p "BlockSize (Seq length) [$seq_length]: " new_seq_length
read -p "Examples [$examples]: " new_examples
read -p "Epochs [$epochs]: " new_epochs
read -p "Learning rate [$lr]: " new_lr
read -p "Training algorithm [$algo]: 'Adadelta' 'Adam' 'AdamW' 'SparseAdam' 'Adamax' 'ASGD' 'NAdam' 'RAdam' 'RMSprop' 'Rprop' 'SGD': " new_algo
read -p "Batch size [$batches]: " new_batches
read -p "Local Model [$local_model]: " new_local_model
read -p "Input model [$model_input]: " new_model_input
read -p "Output model V1.xx [$model_output]: " new_model_output
read -p "Num GPU (in cuda) [$numgpu]: " new_numgpu

# Set the input values to the new values entered by the user, or to the last used values if the user didn't e>
seq_length=${new_seq_length:-$seq_length}
examples=${new_examples:-$examples}
epochs=${new_epochs:-$epochs}
lr=${new_lr:-$lr}
algo=${new_algo:-$algo}
batches=${new_batches:-$batches}
local_model=${new_local_model:-$local_model}
model_input=${new_model_input:-$model_input}
model_output=${new_model_output:-$model_output}
numgpu=${new_numgpu:-$numgpu}

# Store the input values in the file
echo $seq_length $examples $epochs $lr $algo $batches $local_model $model_input $model_output $numgpu > $INPUT_FILE

# Run the script using the input values
clear && CUDA_VISIBLE_DEVICES=$numgpu accelerate launch /root/clm_model_tuning/finetune_using_clm_beta_2.py \
dataset.block_size=$seq_length \
dataset.num_batches=32 \
dataset.num_examples=$examples \
training.num_epochs=$epochs \
training.train_batch_size=$batches \
training.eval_batch_size=$batches \
training.eps=0.0032 \
training.learning_rate=$lr \
training.momentum=0.96 \
training.weight_decay=0.65 \
training.max_train_steps=500 \
training.lr_scheduler=cosine_with_restarts \
training.algorithm=$algo \
dataset.use_ipfs=True \
model.name=$model_input \
output_dir=$model_output
