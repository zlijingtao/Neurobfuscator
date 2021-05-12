cd ../trace_gen
batch_size=1
#perform 10 times for testing purpose, perform at least 4000 times to train a good predictor
MAX_ITER=10
opt_level=3
file_name=torch_tvm_prof
executor_name=torch_tvm_execute

#CIFAR-10

input_channel=3
input_features=3072
num_classes=10

for (( N=0; N<=${MAX_ITER}; N++ ))
do
  output_file=complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features}_seed_${N}.csv
  python func_gen_cifar.py --output_file ${output_file} --input_features ${input_features} --input_channel ${input_channel} --batch_size ${batch_size} --num_classes ${num_classes} --random_seed ${N} --file_name ${file_name} --opt_level ${opt_level}
done

#gen seq dataset

python trace_dataset_gen.py --dataset_type reduced --selection complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features} --prefix_output cifar10_train_data
python trace_dataset_gen.py --dataset_type full --selection complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features} --prefix_output cifar10_train_data
python trace_dataset_gen.py --dataset_type time_only --selection complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features} --prefix_output cifar10_train_data

#gen dim dataset

selection=complex_batch_1_nclass_10_infeat_3072
image_dim=32

prefix_output=cifar10_train_data

type=reduced
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}

type=full
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}

type=time_only
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}
