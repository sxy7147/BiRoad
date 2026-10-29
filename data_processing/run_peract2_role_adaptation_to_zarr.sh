#!/usr/bin/env bash
set -e

# All paths below are relative to the project root.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Episodes per task: use 50/50 or 95/5, for example (both must be positive).
# The same counts apply to train and val; enough raw episodes must be available.
original_count=50
role_swapped_count=50
root=./data/raw
tgt=./data/zarr_datasets/ratio_${original_count}_${role_swapped_count}
im_size=128

tasks=(
    bimanual_handover_item_easy_random:${original_count}
    bimanual_handover_item_easy_symmetric_random:${role_swapped_count}
    bimanual_pick_laptop_random:${original_count}
    bimanual_pick_laptop_symmetric_random:${role_swapped_count}
    bimanual_pick_plate_random:${original_count}
    bimanual_pick_plate_symmetric_random:${role_swapped_count}
    bimanual_put_bottle_in_fridge:${original_count}
    bimanual_put_bottle_in_fridge_symmetric:${role_swapped_count}
    bimanual_put_item_in_drawer_random:${original_count}
    bimanual_put_item_in_drawer_symmetric_random:${role_swapped_count}
    bimanual_sweep_to_dustpan_random:${original_count}
    bimanual_sweep_to_dustpan_symmetric_random:${role_swapped_count}
    bimanual_take_tray_out_of_oven:${original_count}
    bimanual_take_tray_out_of_oven_symmetric:${role_swapped_count}
    coordinated_take_shoes_out_of_box:${original_count}
    coordinated_take_shoes_out_of_box_symmetric:${role_swapped_count}
)


python -m data_processing.peract2_to_zarr \
    --root "$root" \
    --tgt "${tgt}" \
    --split train val \
    --tasks "${tasks[@]}" \
    --im_size "$im_size"
