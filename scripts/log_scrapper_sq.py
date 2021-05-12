
import os
def ga_file_searcher(dir_path):
    file_dict = {}
    for filename in os.listdir(dir_path):
        if "_genetic_algorithm_model" in filename:
            with open(dir_path + filename) as f:
                lines = f.readlines()
            highest_score = 0.0
            output_LER = 0.0
            output_latency = 0.0
            time_budget = "NA"
            for line in lines:
                if "Time Budget" in line:
                    time_budget = "{:1.2f}".format(float(line.split("Time Budget is ")[-1]))

                if "Fitness Score: " in line and "Final" not in line:
                    score = float(line.split(";")[0].split("Fitness Score: ")[-1])
                    LER = float(line.split(";")[1].split("avgLER: ")[-1])
                    latency = float(line.split(";")[2].split("Cycle: ")[-1])
                    if score >= highest_score:
                        highest_score = score
                        output_LER = LER
                        output_latency = latency
            print(filename + "(TB - {}) : {:1.0f}, {:1.3f}, {:1.0f}".format(time_budget, highest_score, output_LER, output_latency))
    # return newest_file
def dim_ga_file_searcher(dir_path, model_name):
    file_dict = {}
    for filename in os.listdir(dir_path):
        if "dim_model{}".format(model_name) in filename:
            with open(dir_path + filename) as f:
                lines = f.readlines()
            highest_score = 0.0
            output_DER = 0.0
            output_latency = 0.0
            time_budget = "NA"
            for line in lines:
                if "Time Budget" in line:
                    time_budget = "{:1.2f}".format(float(line.split("Time Budget is ")[-1]))

                if "Fitness Score: " in line and "Final" not in line:
                    score = float(line.split(";")[0].split("Fitness Score: ")[-1])
                    DER = float(line.split(";")[1].split("avgDER: ")[-1])
                    latency = float(line.split(";")[2].split("Cycle: ")[-1])
                    if score >= highest_score:
                        highest_score = score
                        output_DER = DER
                        output_latency = latency
            print(filename + "(TB - {}) : {:1.0f}, {:1.3f}, {:1.0f}".format(time_budget, highest_score, output_DER, output_latency))

if __name__ == '__main__':
    dim_ga_file_searcher("../dim_obfuscator/", "2")