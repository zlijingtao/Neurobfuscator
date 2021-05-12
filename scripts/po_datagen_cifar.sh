cd ../trace_gen

selection=complex_batch_1_nclass_10_infeat_3072
image_dim=32

prefix_output=PL_train_data

type=reduced
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}

type=full
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}

type=time_only
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}
