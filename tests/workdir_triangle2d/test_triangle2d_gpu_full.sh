#!/usr/bin/env bash

set -e
CUDA_VISIBLE_DEVICES=""

for dir in "./models/triangle_dummy" "./data/triangle2d_316_dummy"; do
    if [ -d "$dir" ]; then
        rm -r "$dir"
    fi
done

echo "#####"
echo "Generate validation data:"
echo "#####"

PYTHONPATH=../../../tf_neiss:$PYTHONPATH python -u ../../../tf_neiss/input_fn/input_fn_2d/data_gen_2dt/generate_train_data_2d_triangle_TFR.py \
    --to_log_file --mode "val" --data_id triangle2d_32_dummy

echo "#####"
echo "Train model on validation data:"
echo "#####"
echo ""
PYTHONPATH=../../../tf_neiss:$PYTHONPATH python -u ../../../tf_neiss/trainer/trainer_types/trainer_2dt/trainer_triangle2d.py \
    --train_lists "./lists/triangle2d_32_dummy_val.lst" \
    --val_list "./lists/triangle2d_32_dummy_val.lst" \
    --checkpoint_dir "models/triangle_dummy" \
    --data_len 32 \
    --epochs 5 \
    --gpu_devices 0 \

echo "#####"
echo "Test model on validation data:"
echo "#####"
echo ""
PYTHONPATH=../../../tf_neiss:$PYTHONPATH python -u ../../../tf_neiss/trainer/lav_types/lav_triangle2d.py \
    --model_dir models/triangle_dummy \
    --val_list \
    ./lists/triangle2d_32_dummy_val.lst \
    --data_len 32 \
    --val_batch_size \
    50 \
    --batch_limiter \
    1 \
    --plot True \
    --gpu_devices 0 \

echo "Copy result to workdir"
cp ./models/triangle_dummy/plot_summary.pdf ./
echo "Done."


echo "#####"
echo "Clean up..."
echo "#####"
echo ""
for dir in "./models/triangle_dummy" "./data/triangle2d_316_dummy"; do
    if [ -d "$dir" ]; then
        rm -r "$dir"
    fi
done
echo "Done."
