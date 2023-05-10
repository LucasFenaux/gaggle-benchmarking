export CUDA_VISIBLE_DEVICES=0

for pop_size in 10 32 100 200 320 1000
do
    python3 gaggle_experiment.py --config gaggle_mnist.yml --population_size $pop_size
    python3 leap.py --population_size $pop_size
    python3 pygad.py --population_size $pop_size
done
