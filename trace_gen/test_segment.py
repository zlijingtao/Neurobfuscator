from trace_dataset_gen import trace_csv_seg

# trace_file = "None", fuse = True, trace_type = "tvm", no_repeat = False

dir = "trace/"
# trace_file = "batch_1_nclass_10_infeat_3072_seed_100.csv"
trace_file = "complex_batch_1_nclass_10_infeat_3072_seed_2377.csv"

print(trace_csv_seg(dir + trace_file))