#!/bin/bash

BASEPATH="/home/daniel/DLRT-Net-main/cifar10/results/resnet20/resnet20_cifar10_baseline_0.0__best_weights.pt"

for X in 0.04 0.08 0.12 0.16 0.2; do
    MODELPATH="/home/daniel/DLRT-Net-main/cifar10/results/resnet20/resnet20_cifar10_${X}_best_weights.pt"
    OUTPUT_DIR="./output_guess/resnet20_cifar10_${X}"

    python cam.py --device "cuda" \
        --method "gradcam" \
        --image-path "./test_images/" \
        --model-path "$MODELPATH" \
        --output-dir "$OUTPUT_DIR" \
        --low-rank \

done

python cam.py --device "cuda" \
        --method "gradcam" \
        --image-path "./test_images/" \
        --model-path "$BASEPATH" \
        --output-dir "./output_guess/"