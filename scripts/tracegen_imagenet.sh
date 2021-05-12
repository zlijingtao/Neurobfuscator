cd ../trace_gen

batch_size=1
#sweep 2000 times
MAX_ITER=2000
opt_level=3
file_name=torch_tvm_prof
executor_name=torch_tvm_execute

#ImageNet

input_channel=3
input_features=150528
num_classes=10

for (( N=0; N<=${MAX_ITER}; N++ ))
do
  output_file=complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features}_seed_${N}.csv
  python func_gen_imagenet.py --output_file ${output_file} --input_features ${input_features} --input_channel ${input_channel} --batch_size ${batch_size} --num_classes ${num_classes} --random_seed ${N} --file_name ${file_name} --opt_level ${opt_level}
done

for (( N=2000; N<=4000; N++ ))
do
  output_file=complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features}_seed_${N}.csv
  python func_gen_imagenet_depth.py --output_file ${output_file} --input_features ${input_features} --input_channel ${input_channel} --batch_size ${batch_size} --num_classes ${num_classes} --random_seed ${N} --file_name ${file_name} --opt_level ${opt_level}
done

#gen seq dataset

python trace_dataset_gen.py --dataset_type full --selection complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features} --prefix_output imagenet_train_data
python trace_dataset_gen.py --dataset_type time_only --selection complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features} --prefix_output imagenet_train_data
python trace_dataset_gen.py --dataset_type reduced --selection complex_batch_${batch_size}_nclass_${num_classes}_infeat_${input_features} --prefix_output imagenet_train_data

#gen dim dataset

selection=complex_batch_1_nclass_1000_infeat_150528
image_dim=224

prefix_output=imagenet_train_data

type=reduced
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}

type=full
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}

type=time_only
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}
