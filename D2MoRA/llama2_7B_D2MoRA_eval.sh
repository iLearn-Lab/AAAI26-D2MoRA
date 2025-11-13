# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

export model_path='meta-llama/Llama-2-7B-hf'
export weight_path='checkpoints/D2MoRA'

export device_id=0

CUDA_VISIBLE_DEVICES=$device_id nohup python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter D2MoRA \
    --dataset piqa \
    --base_model $model_path \
    --batch_size 1 \
    --lora_weights $weight_path|tee $weight_path/piqa.txt &

wait $!

CUDA_VISIBLE_DEVICES=$device_id nohup python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter D2MoRA \
    --dataset boolq \
    --base_model $model_path \
    --batch_size 1 \
    --lora_weights $weight_path|tee $weight_path/boolq.txt &

wait $!

CUDA_VISIBLE_DEVICES=$device_id nohup python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter D2MoRA \
    --dataset social_i_qa \
    --base_model $model_path \
    --batch_size 1 \
    --lora_weights $weight_path|tee $weight_path/social_i_qa.txt &

wait $!

CUDA_VISIBLE_DEVICES=$device_id nohup python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter D2MoRA \
    --dataset winogrande \
    --base_model $model_path \
    --batch_size 1 \
    --lora_weights $weight_path|tee $weight_path/winogrande.txt &

wait $!

CUDA_VISIBLE_DEVICES=$device_id nohup python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter D2MoRA \
    --dataset ARC-Challenge \
    --base_model $model_path \
    --batch_size 1 \
    --lora_weights $weight_path|tee $weight_path/ARC-Challenge.txt &

wait $!

CUDA_VISIBLE_DEVICES=$device_id nohup python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter D2MoRA \
    --dataset ARC-Easy \
    --base_model $model_path \
    --batch_size 1 \
    --lora_weights $weight_path|tee $weight_path/ARC-Easy.txt &

wait $!

CUDA_VISIBLE_DEVICES=$device_id nohup python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter D2MoRA \
    --dataset openbookqa \
    --base_model $model_path \
    --batch_size 1 \
    --lora_weights $weight_path|tee $weight_path/openbookqa.txt &

wait $!

CUDA_VISIBLE_DEVICES=$device_id nohup python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter D2MoRA \
    --dataset hellaswag \
    --base_model $model_path \
    --batch_size 1 \
    --lora_weights $weight_path|tee $weight_path/hellaswag.txt &