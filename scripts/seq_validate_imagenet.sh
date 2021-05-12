cd ../seq_predictor

choice=obfuscator

normalize=smart


num_layers=1

num_hidden_list=(512 256 128)

dataset_type_list=("full")

restore_step=149

for val1 in ${num_hidden_list[*]}; do
    for val2 in ${dataset_type_list[*]}; do
        num_hidden=$val1
        dataset_type=$val2
        model_name=deepsniffer_LSTM_both_autotvm_${normalize}_imagenet_${num_hidden}_imagenet

        if [[ "${dataset_type}" == "reduced" ]]; then
            model_path="./${choice}/predictor/logs_${model_name}"
            dataset="./${choice}/dataset/imagenet_train_data_dict.pickle"
        elif [[ "${dataset_type}" == "full" ]]; then
            model_path="./${choice}/predictor/logs_full_${model_name}"
            dataset="./${choice}/dataset/imagenet_train_data_dict_full.pickle"
        elif [[ "${dataset_type}" == "time_only" ]]; then
            model_path="./${choice}/predictor/logs_timeonly_${model_name}"
            dataset="./${choice}/dataset/imagenet_train_data_dict.pickle"
        fi

        python validate.py --dataset ${dataset} --model_path ${model_path} --train_type ${dataset_type} --restore_step ${restore_step} --normalize ${normalize} --num_hidden ${num_hidden} --num_layers ${num_layers}

    done
done