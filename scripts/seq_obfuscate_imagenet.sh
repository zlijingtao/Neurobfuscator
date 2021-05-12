cd ../seq_obfuscator

run_option=genetic_algorithm

batch_size=1

input_features=150528

predict_type=full

normalize=smart

model_name_base=deepsniffer_LSTM_both_autotvm_smart_imagenet
model_name_afterfix=imagenet
num_predictor_model=3
restore_step=49
reward_type=divide_square_residue_offseted

# your_model_id=
# your_budget=
# budget_list=(${your_budget})
# nn_id_list=(${your_model_id})
# n_pop=16
# n_generation=20
#Run below for testing, run above for bagging of 3 LSTM predictors
budget_list=(0.20)
nn_id_list=(10)
n_pop=4
n_generation=2

for budget in ${budget_list[*]}
do
    for nn_id in ${nn_id_list[*]}
    do
    python seq_obfuscate.py --batch_size ${batch_size} --input_features ${input_features} --nn_id ${nn_id} \
    --predict_type ${predict_type} --normalize ${normalize} --model_name_base ${model_name_base} \
    --model_name_afterfix ${model_name_afterfix} --num_predictor_model ${num_predictor_model} \
    --restore_step ${restore_step} --reward_type ${reward_type} --budget ${budget} --run_option ${run_option} --has_skip \
    --forbid_kerneladd --forbid_dummy --forbid_prune --continue_from_newest
    done
done
