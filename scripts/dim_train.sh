#!/usr/bin/env sh
#Use to train dimension obfuscation - random forest regression model
cd ../dim_predictor

algorithm=rf
dataset_type_list=("full" "reduced" "timeonly")
target_list=("TargetIC" "TargetOC" "TargetKernel" "TargetStride" "TargetPad")

#Use option 3 to Test the model on validation set
# option=3

#Use option 0 to Train the model on Training set
option=0
n_estimators=100
min_samples_split=30
for val2 in ${dataset_type_list[*]}; do
    for val3 in ${target_list[*]}; do
    dataset_type=$val2
    target=$val3
    if [[ ${algorithm} == "rf" ]]; then
        python dim_train.py --dataset_type ${dataset_type} --target ${target} --option ${option} \
        --n_estimators ${n_estimators} --min_samples_split ${min_samples_split}
    fi
    if [[ ${algorithm} == "xgb" ]]; then
        python dim_train_xgb.py --dataset_type ${dataset_type} --target ${target} --option ${option} \
        --n_estimators ${n_estimators} --min_samples_split ${min_samples_split}
    fi
    done
done
