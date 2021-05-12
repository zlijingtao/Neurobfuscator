#!/bin/bash
cd ../seq_obfuscator
#Use to generate tuner test figure showing xgb characteristics do not change much.

nn_id=3

batch_size=1

input_features=3072

predict_type=full

predictor_name=deepsniffer_LSTM_both_autotvm_smart

num_hidden=128

tuner_list=("xgb")

n_trial_list=(200 400 800)
for val1 in ${tuner_list[*]}; do
    for val2 in ${n_trial_list[*]}; do
        tuner=$val1
        n_trial=$val2
        out_name=${val1}_${val2}
        log_file=autotvm_nnid_${nn_id}_obf_${tuner}_${n_trial}.txt
        python torch_relay_obfuscate.py --model_id ${nn_id} --input_features ${input_features} \
        --tuner ${tuner} --n_trial ${n_trial} --run_style test_tuner --out_name ${out_name}\
        --predict_type ${predict_type} --predictor_name ${predictor_name} --num_hidden ${num_hidden}  2>&1 | tee ./obf_tmp_file/${log_file}

    done
done
