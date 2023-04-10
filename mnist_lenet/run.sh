export CUDA_VISIBLE_DEVICES=0

for pop_size in 10 32 100 320 1000 3200
do
    python3 gaggle.py --config gaggle_mnist.yml --population_size $pop_size
done