import numpy as np

random_seed = 500

# file_path = "./trace/complex_batch_1_nclass_1000_infeat_150528_seed_{}.npy".format(random_seed)
file_path = "./trace/complex_batch_1_nclass_10_infeat_3072_seed_{}.npy".format(random_seed)

# mstring_path = "./trace/complex_batch_1_nclass_1000_infeat_150528_seed_{}.mstring".format(random_seed)
mstring_path = "./trace/complex_batch_1_nclass_10_infeat_3072_seed_{}.mstring".format(random_seed)

npy_array = np.load(file_path)

print("label file is: ", npy_array)

my_file = open(mstring_path, "r")

content_list = my_file.readlines()

print("model name is: ", content_list)