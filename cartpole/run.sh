export CUDA_VISIBLE_DEVICES=0


pop_size=100


python3 gaggle.py --config gaggle_cartpole.yml --population_size $pop_size --model_size "tiny" --device cpu
python3 leap.py --population_size $pop_size --model_size "tiny" --device cpu
python3 pygad.py --population_size $pop_size --model_size "tiny" --device cpu

for model_size in "small" "medium"
do
  python3 gaggle.py --config gaggle_cartpole.yml --population_size $pop_size --model_size $model_size --device cuda
  python3 leap.py --population_size $pop_size --model_size $model_size --device cuda
  python3 pygad.py --population_size $pop_size --model_size $model_size --device cuda
done

python3 gaggle.py --config gaggle_cartpole.yml --population_size $pop_size --model_size "large" --device cuda --model_name "large_dqn"
python3 leap.py --population_size $pop_size --model_size "large" --device cuda
python3 pygad.py --population_size $pop_size --model_size "large" --device cuda


#for pop_size in 10 32 100 320 1000
#do
#  python3 gaggle.py --config gaggle_cartpole.yml --population_size $pop_size --model_size "tiny" --device cpu
#  python3 leap.py --population_size $pop_size --model_size "tiny" --device cpu
#  python3 pygad.py --population_size $pop_size --model_size "tiny" --device cpu
#done

#for pop_size in 10 32 100 320 1000
#do
#  for model_size in "small" "medium"
#  do
#    python3 gaggle.py --config gaggle_cartpole.yml --population_size $pop_size --model_size $model_size --device cuda
#    python3 leap.py --population_size $pop_size --model_size $model_size --device cuda
#    python3 pygad.py --population_size $pop_size --model_size $model_size --device cuda
#  done
#done
