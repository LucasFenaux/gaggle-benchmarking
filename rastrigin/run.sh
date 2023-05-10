export CUDA_VISIBLE_DEVICES=0

for dim in 10 100 300 500 800 1000
do
    python3 gaggle_experiment.py --config gaggle_rastrigin.yml --individual_size $dim
    python3 leap.py --dimension $dim
    python3 pygad.py --dimension $dim
done