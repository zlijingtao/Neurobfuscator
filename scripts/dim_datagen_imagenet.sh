cd ../trace_gen

selection=complex_batch_1_nclass_1000_infeat_150528
image_dim=224

prefix_output=imagenet_train_data

type=reduced
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}

type=full
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}

type=time_only
python classify_dataset_gen.py --dataset_type ${type} --selection ${selection} --prefix_output ${prefix_output} --orig_img_dim ${image_dim}
