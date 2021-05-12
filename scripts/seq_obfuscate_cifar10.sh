cd ../seq_obfuscator

run_option=genetic_algorithm

batch_size=1

input_features=3072

normalize=smart

model_name_base=deepsniffer_LSTM_both_autotvm_smart
model_name_afterfix=cifar10
reward_type=divide_square_residue_offseted

# budget_list=(0.01 0.02 0.05 0.10 0.20)
# nn_id_list=(4 5 6 7)
# restore_step=149
# num_predictor_model=5
# predict_type=all
# n_pop=16
# n_generation=20
#Run below for testing, run above for bagging of 15 LSTM predictors
budget_list=(0.20)
nn_id_list=(4)
restore_step=49
num_predictor_model=1
predict_type=full
n_pop=4
n_generation=2

for budget in ${budget_list[*]}
do
    for nn_id in ${nn_id_list[*]} #Run VGG-11, resnet-20, VGG-13, resnet-32
    do
    python seq_obfuscate.py --batch_size ${batch_size} --input_features ${input_features} --nn_id ${nn_id} \
    --predict_type ${predict_type} --normalize ${normalize} --model_name_base ${model_name_base} \
    --model_name_afterfix ${model_name_afterfix} --num_predictor_model ${num_predictor_model} \
    --restore_step ${restore_step} --reward_type ${reward_type} --budget ${budget} --run_option ${run_option} --has_skip \
    --forbid_kerneladd --forbid_dummy --forbid_prune --n_pop ${n_pop} --n_generation ${n_generation}
    done
done
