#%% Use to scrap hp experiment prune. get all reward and cycle in list format.
import os
import csv
# import matplotlib.pyplot as plt
import numpy
def ga_file_searcher(dir_path, datatime):
    file_dict = {}
    reward_list = []
    cycle_list = []
    for filename in os.listdir(dir_path):
        if "dim_model" in filename and datatime in filename:
            with open(dir_path + filename) as f:
                lines = f.readlines()
                clean_cycle = 0
                for line in lines:
                    if "Clean Cycle is: " in line:
                        clean_cycle = float(line.split("Clean Cycle is: ")[1].split(",")[0])
                if clean_cycle == 0:
                    print("Clean cycle not found!")
                    return None, None

                for line in lines:
                    reward = 0
                    cycle = 0


                    if "Current Reward: " in line:
                        reward = float(line.split("Current Reward: ")[1].split(",")[0])
                    if "Cycle: " in line:
                        cycle = float(line.split("Cycle: ")[1].split(",")[0])/clean_cycle
                    if cycle != 0 and reward != 0:
                        reward_list.append(reward)
                        cycle_list.append(cycle)
    sort = True
    if sort:
        reward = numpy.array(reward_list)
        cycle = numpy.array(cycle_list)
        inds = cycle.argsort()
        cycle_list = cycle[inds].tolist()
        reward_list = reward[inds].tolist()
    return reward_list, cycle_list




#%%
if __name__ == '__main__':
    reward_list, cycle_list = ga_file_searcher("../dim_obfuscator/", "05032030")
    with open('../dim_obfuscator/log_scrapper.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(cycle_list)
        write.writerow(reward_list)