#!/bin/bash
# Download 100M cube-quadruple datasets

DATA_DIR="/scratch/hliuson/.ogbench/data"
BASE_URL="https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0"

mkdir -p "$DATA_DIR"

echo "Downloading cube-quadruple-play-100m-v0 train dataset..."
wget -c "${BASE_URL}/cube-quadruple-play-100m-v0.npz" -O "${DATA_DIR}/cube-quadruple-play-100m-v0.npz"

echo "Downloading cube-quadruple-play-100m-v0 validation dataset..."
wget -c "${BASE_URL}/cube-quadruple-play-100m-v0-val.npz" -O "${DATA_DIR}/cube-quadruple-play-100m-v0-val.npz"

echo "Download complete!"
ls -la "${DATA_DIR}/"*100m*
