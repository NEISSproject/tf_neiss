#!/usr/bin/env bash

set -e


for dir in "./models/triangle_dummy" "./data/triangle2d_316_dummy"; do
    if [ -d "$dir" ]; then
        rm -r "$dir"
    fi
done

echo "#####"
echo "Generate validation data:"
echo "#####"

python -u generate_train_data_2d_triangle_TFR.py \
    --to_log_file --mode "val" --data_id triangle2d_32_dummy

echo "#####"
echo "Train model on validation data:"
echo "#####"
echo ""
python -u trainer_triangle2d.py \
    --train_lists "./lists/triangle2d_32_dummy_val.lst" \
    --val_list "./lists/triangle2d_32_dummy_val.lst" \
    --checkpoint_dir "models/triangle_dummy" \
    --data_len 32 \
    --epochs 5 \

echo "#####"
echo "Test model on validation data:"
echo "#####"
echo ""
python -u lav_triangle2d.py \
    --model_dir models/triangle_dummy \
    --val_list \
    ./lists/triangle2d_32_dummy_val.lst \
    --data_len 32 \
    --val_batch_size \
    50 \
    --batch_limiter \
    1 \
    --plot True \

echo "Copy result to reports"
cp ./models/triangle_dummy/plot_summary.pdf ./reports
echo "Done."


echo "#####"
echo "Clean up..."
echo "#####"
echo ""
for dir in "./models/triangle_dummy" "./data/triangle2d_316_dummy" "data/synthetic_data/triangle2d_32_dummy"; do
    if [ -d "$dir" ]; then
        rm -r "$dir"
    fi
done

for file in "./lists/triangle2d_32_dummy_val.lst" "./data/triangle2d_316_dummy"; do
    if [ -f "$file" ]; then
        rm "$file"
    fi
done
echo "Done."
