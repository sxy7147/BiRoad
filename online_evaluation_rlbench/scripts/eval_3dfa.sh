exp=3dfa_95_5
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

use_sym_anti_decomp=false
sym_anti_direct_update=false
sym_anti_decomp_placement=multi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra cuda_ids <<< "$CUDA_VISIBLE_DEVICES"
else
    num_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
    cuda_ids=($(seq 0 $((num_gpus - 1))))
fi
num_gpus=${#cuda_ids[@]}

if ((num_gpus == 0)); then
    echo "No GPU found. Please set CUDA_VISIBLE_DEVICES or check nvidia-smi." >&2
    exit 1
fi

# Run evaluation workers on a single GPU by default. If CUDA_VISIBLE_DEVICES
# exposes multiple GPUs, use the first one unless EVAL_GPU is set.
eval_gpu=${EVAL_GPU:-${cuda_ids[0]}}
cuda_ids=($eval_gpu)
num_gpus=1

n_workers=${N_WORKERS:-1}
if ! [[ "$n_workers" =~ ^[0-9]+$ ]] || ((n_workers <= 0)); then
    echo "N_WORKERS must be a positive integer, got: $n_workers" >&2
    exit 1
fi

# Testing arguments
B=128  # Per-GPU batch size used during training.
checkpoint=./train_logs/$exp/denoise3d-Peract2_3dfront_3dwrist-C120-B${B}-lr1e-4-constant-H3-rectified_flow/best.pth
checkpoint_alias=eval_on_mixed_data

max_tries=2
max_steps=25
headless=true
collision_checking=false
seed=0

# Dataset arguments
data_dir=./data/test_raw
dataset=Peract2_3dfront_3dwrist
image_size=128,128
num_demos=50
save_video=true
video_resolution="${VIDEO_RESOLUTION:-128,128}"
video_camera="${VIDEO_CAMERA:-front}"
video_freq=1

# Model arguments
model_type=denoise3d
bimanual=true
prediction_len=1

backbone=clip
fps_subsampling_factor=4

embedding_dim=120
num_attn_heads=8
num_vis_instr_attn_layers=3
num_history=3

num_shared_attn_layers=4
relative_action=false
rotation_format=quat_xyzw
denoise_timesteps=5
denoise_model=rectified_flow

export HF_ENDPOINT="https://hf-mirror.com"
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

eval_root=eval_logs/$exp/$checkpoint_alias/seed$seed
num_ckpts=${#tasks[@]}

echo "Evaluating $num_ckpts tasks with n_workers=$n_workers over $num_gpus GPU(s): ${cuda_ids[*]}"

running_jobs=0
for ((i=0; i<$num_ckpts; i++)); do
    gpu_idx=$((i % num_gpus))
    cuda_id=${cuda_ids[$gpu_idx]}

    CUDA_VISIBLE_DEVICES=$cuda_id xvfb-run -a python online_evaluation_rlbench/evaluate_policy.py \
        --checkpoint $checkpoint \
        --task ${tasks[$i]} \
        --max_tries $max_tries \
        --max_steps $max_steps \
        --headless $headless \
        --collision_checking $collision_checking \
        --seed $seed \
        --data_dir $data_dir \
        --dataset $dataset \
        --image_size $image_size \
        --output_file $eval_root/${tasks[$i]}/eval.json  \
        --save_video $save_video \
        --video_dir $eval_root/${tasks[$i]}/videos \
        --video_resolution $video_resolution \
        --video_camera $video_camera \
        --model_type $model_type \
        --bimanual $bimanual \
        --prediction_len $prediction_len \
        --backbone $backbone \
        --fps_subsampling_factor $fps_subsampling_factor \
        --embedding_dim $embedding_dim \
        --num_attn_heads $num_attn_heads \
        --num_vis_instr_attn_layers $num_vis_instr_attn_layers \
        --num_history $num_history \
        --num_shared_attn_layers $num_shared_attn_layers \
        --relative_action $relative_action \
        --rotation_format $rotation_format \
        --denoise_timesteps $denoise_timesteps \
        --denoise_model $denoise_model \
        --use_sym_anti_decomp $use_sym_anti_decomp \
        --sym_anti_direct_update $sym_anti_direct_update \
        --sym_anti_decomp_placement $sym_anti_decomp_placement \
        --video_freq $video_freq \
        --num_demos $num_demos &

    running_jobs=$((running_jobs + 1))
    if ((running_jobs >= n_workers)); then
        wait -n
        running_jobs=$((running_jobs - 1))
    fi
done


wait

python online_evaluation_rlbench/collect_results.py \
    --folder $eval_root/
