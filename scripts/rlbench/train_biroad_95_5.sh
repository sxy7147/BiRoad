main_dir=BiRoad_95_5
# same-language 95/5 split + symAnti decomposition

DATA_PATH=./data/

train_data_dir=$DATA_PATH/zarr_datasets/ratio_95_5/train.zarr
eval_data_dir=$DATA_PATH/zarr_datasets/ratio_50_50/val.zarr
train_instructions=$DATA_PATH/zarr_datasets/ratio_95_5/instructions.json
val_instructions=$DATA_PATH/zarr_datasets/ratio_50_50/instructions.json

# sym-anti decomposition related
use_sym_anti_decomp=true
sym_anti_direct_update=false
sym_anti_decomp_placement=multi


if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_id=$CUDA_VISIBLE_DEVICES
    ngpus=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
    ngpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
    gpu_id=$(seq -s, 0 $((ngpus - 1)))
fi

if ((ngpus <= 0)); then
    echo "No GPU found. Please set CUDA_VISIBLE_DEVICES or check nvidia-smi." >&2
    exit 1
fi

B=$((256 / ngpus))

tasks=(
    bimanual_handover_item_easy_random
    bimanual_handover_item_easy_symmetric_random
    bimanual_pick_laptop_random
    bimanual_pick_laptop_symmetric_random
    bimanual_pick_plate_random
    bimanual_pick_plate_symmetric_random
    bimanual_put_bottle_in_fridge
    bimanual_put_bottle_in_fridge_symmetric
    bimanual_put_item_in_drawer_random
    bimanual_put_item_in_drawer_symmetric_random
    bimanual_sweep_to_dustpan_random
    bimanual_sweep_to_dustpan_symmetric_random
    bimanual_take_tray_out_of_oven
    bimanual_take_tray_out_of_oven_symmetric
    coordinated_take_shoes_out_of_box
    coordinated_take_shoes_out_of_box_symmetric
)

all_tasks=(
    bimanual_handover_item_easy_random
    bimanual_handover_item_easy_symmetric_random
    bimanual_open_bottle_random
    bimanual_open_bottle_symmetric_random
    bimanual_pick_laptop_random
    bimanual_pick_laptop_symmetric_random
    bimanual_pick_plate_random
    bimanual_pick_plate_symmetric_random
    bimanual_put_bottle_in_fridge
    bimanual_put_bottle_in_fridge_symmetric
    bimanual_put_item_in_drawer_random
    bimanual_put_item_in_drawer_symmetric_random
    bimanual_sweep_to_dustpan_random
    bimanual_sweep_to_dustpan_symmetric_random
    bimanual_take_tray_out_of_oven
    bimanual_take_tray_out_of_oven_symmetric
    coordinated_take_shoes_out_of_box
    coordinated_take_shoes_out_of_box_symmetric
)

custom_img_size=128

dataset=Peract2_3dfront_3dwrist
num_workers=8
# B=64  # we used 64 but you can use as low as 16 without much performance drop - it's much faster
B_val=64
chunk_size=1  # trajectory_length
memory_limit=16  # this means 8GB CPU RAM per worker per GPU,
# but it will never reach that, because these datasets are small
# reduce this if you can't allocate more than 96GB of CPU memory

# Training/testing arguments
val_freq=10000
num_best_checkpoints=3
eval_only=false
lr=1e-4
backbone_lr=1e-6  # doesn't matter when we don't finetune
lr_scheduler=constant
wd=1e-10
train_iters=500000
use_compile=false  # much faster, but sometimes unstable
use_ema=false
lv2_batch_size=1  # you can increase this and divide B equally, speed/accuracy tradeoff

# Model arguments, change (some of) these for new architectures
model_type=denoise3d
bimanual=true
keypose_only=true
pre_tokenize=true
workspace_normalizer_buffer=0.05

backbone=clip
finetune_backbone=false
finetune_text_encoder=false
fps_subsampling_factor=4

C=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

run_log_dir=$model_type-$dataset-C$C-B$B-lr$lr-$lr_scheduler-H$num_history-$denoise_model
checkpoint=train_logs/${main_dir}/${run_log_dir}/last.pth

export HF_ENDPOINT="https://hf-mirror.com"
# unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME="$(realpath ../hf_cache)"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
export CLIP_CACHE_DIR="$HUGGINGFACE_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_ENABLE_HF_TRANSFER


CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node $ngpus --master_port $((RANDOM % 30000 + 20000)) \
    main.py \
    --train_data_dir $train_data_dir \
    --eval_data_dir $eval_data_dir \
    --custom_img_size $custom_img_size \
    --tasks "${tasks[@]}" \
    --all_tasks "${all_tasks[@]}" \
    --train_instructions $train_instructions \
    --val_instructions $val_instructions \
    --dataset $dataset \
    --num_workers $num_workers \
    --batch_size $B \
    --batch_size_val $B_val \
    --chunk_size $chunk_size \
    --memory_limit $memory_limit \
    --exp_log_dir $main_dir \
    --run_log_dir ${run_log_dir} \
    --checkpoint $checkpoint \
    --val_freq $val_freq \
    --num_best_checkpoints $num_best_checkpoints \
    --eval_only $eval_only \
    --lr $lr \
    --backbone_lr $backbone_lr \
    --lr_scheduler $lr_scheduler \
    --wd $wd \
    --train_iters $train_iters \
    --use_compile $use_compile \
    --use_ema $use_ema \
    --lv2_batch_size $lv2_batch_size \
    --model_type $model_type \
    --bimanual $bimanual \
    --keypose_only $keypose_only \
    --pre_tokenize $pre_tokenize \
    --backbone $backbone \
    --finetune_backbone $finetune_backbone \
    --finetune_text_encoder $finetune_text_encoder \
    --fps_subsampling_factor $fps_subsampling_factor \
    --embedding_dim $C \
    --num_attn_heads $num_attn_heads \
    --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
    --num_history $num_history \
    --num_shared_attn_layers $num_shared_attn_layers \
    --workspace_normalizer_buffer $workspace_normalizer_buffer \
    --relative_action $relative_action \
    --rotation_format $rotation_format \
    --denoise_timesteps $denoise_timesteps \
    --denoise_model $denoise_model \
    --use_sym_anti_decomp $use_sym_anti_decomp \
    --sym_anti_direct_update $sym_anti_direct_update \
    --sym_anti_decomp_placement $sym_anti_decomp_placement \
