cd ../trace_obfuscate

run_option=genetic_algorithm

batch_size=1

input_features=150528

predict_type=full

normalize=smart

model_name_base=deepsniffer_LSTM_both_autotvm_smart_imagenet
model_name_afterfix=depth
num_predictor_model=3
restore_step=49
reward_type=divide_square_residue_offseted

for budget in 0.01 0.02 0.05 0.10 0.20
do
    for nn_id in 8 9 10 #Run VGG-11, resnet-20, VGG-13, resnet-32
    do
    python run_obf.py --batch_size ${batch_size} --input_features ${input_features} --nn_id ${nn_id} \
    --predict_type ${predict_type} --normalize ${normalize} --model_name_base ${model_name_base} \
    --model_name_afterfix ${model_name_afterfix} --num_predictor_model ${num_predictor_model} \
    --restore_step ${restore_step} --reward_type ${reward_type} --budget ${budget} --run_option ${run_option} --has_skip \
    --forbid_kerneladd --forbid_dummy --forbid_prune --continue_from_newest
    done
done
