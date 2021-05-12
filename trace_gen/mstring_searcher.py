# cnn_64p3p2p1p2_32p3p1p1p0
import os

trace_dir = "./trace/"
include_name = "complex_batch"
for filename in os.listdir(trace_dir):
        # if seed in filename:
        #     seed_ID = int(filename.split("_")[-1].split(".")[0])
        if include_name in filename:
            file_prefix = filename.split('.')[0]
            file_type = filename.split('.')[1]
            if file_type == 'mstring':
                with open(trace_dir + filename) as f:
                    lines = f.readlines()
                for line in lines:
                    if "cnn_64p3p1p1p0_res64" in line:
                        print(filename + ": " + line)
                    # if "cnn_512p3p1p1p0_res512" in line:
                    #     print(filename + ": " + line)