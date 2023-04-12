export CUDA_VISIBLE_DEVICES=0

for dim in 10 32 100 320 1000
do
    python3 gaggle.py --config gaggle_rastrigin.yml --dimension $dim
    python3 leap.py --dimension $dim
    python3 pygad.py --dimension $dim
done