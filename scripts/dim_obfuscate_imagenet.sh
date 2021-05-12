cd ../dim_obfuscator

batch_size=1

input_features=150528

run_option=genetic_algorithm

# your_model_id=
# your_budget=
# model_id=${your_model_id}
# budget_list=(${your_budget})
# n_pop=16
# n_generation=20
#Run below for testing, run above for sweeping experiments
budget_list=(0.20)
n_pop=4
n_generation=2
model_id=1


for budget in ${budget_list[*]}; do

python dim_obfuscate.py --batch_size ${batch_size} --input_features ${input_features} --model_id ${model_id} --budget ${budget} --run_option ${run_option}  --n_pop ${n_pop} --n_generation ${n_generation}

done