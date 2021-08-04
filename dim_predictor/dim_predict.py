import argparse
import time
import numpy as np
import pickle
import pandas as pd

class Dim_predictor(object):
    def __init__(self, path_to_model = "./saved_models", n_estimators = 100, min_samples_split = 30, dataset_type = "full"):
        self.dataset_type = dataset_type
        if self.dataset_type != "all":
            model_path = "{}/RF_{}_minsplit_{}_{}_TargetIC.pickle".format(path_to_model, n_estimators, min_samples_split, dataset_type)
            self.IC_classifier = pickle.load(open(model_path, "rb"))

            model_path = "{}/RF_{}_minsplit_{}_{}_TargetOC.pickle".format(path_to_model, n_estimators, min_samples_split, dataset_type)
            self.OC_classifier = pickle.load(open(model_path, "rb"))

            model_path = "{}/RF_{}_minsplit_{}_{}_TargetKernel.pickle".format(path_to_model, n_estimators, min_samples_split, dataset_type)
            self.Kernel_classifier = pickle.load(open(model_path, "rb"))

            model_path = "{}/RF_{}_minsplit_{}_{}_TargetStride.pickle".format(path_to_model, n_estimators, min_samples_split, dataset_type)
            self.Stride_classifier = pickle.load(open(model_path, "rb"))

            model_path = "{}/RF_{}_minsplit_{}_{}_TargetPad.pickle".format(path_to_model, n_estimators, min_samples_split, dataset_type)
            self.Pad_classifier = pickle.load(open(model_path, "rb"))
        else:
            type_list = ["timeonly", "reduced", "full"]
            self.IC_classifier = []
            self.OC_classifier = []
            self.Kernel_classifier = []
            self.Stride_classifier = []
            self.Pad_classifier = []

            for dtype in type_list:
                model_path = "{}/RF_{}_minsplit_{}_{}_TargetIC.pickle".format(path_to_model, n_estimators, min_samples_split, dtype)
                self.IC_classifier.append(pickle.load(open(model_path, "rb")))

                model_path = "{}/RF_{}_minsplit_{}_{}_TargetOC.pickle".format(path_to_model, n_estimators, min_samples_split, dtype)
                self.OC_classifier.append(pickle.load(open(model_path, "rb")))

                model_path = "{}/RF_{}_minsplit_{}_{}_TargetKernel.pickle".format(path_to_model, n_estimators, min_samples_split, dtype)
                self.Kernel_classifier.append(pickle.load(open(model_path, "rb")))

                model_path = "{}/RF_{}_minsplit_{}_{}_TargetStride.pickle".format(path_to_model, n_estimators, min_samples_split, dtype)
                self.Stride_classifier.append(pickle.load(open(model_path, "rb")))

                model_path = "{}/RF_{}_minsplit_{}_{}_TargetPad.pickle".format(path_to_model, n_estimators, min_samples_split, dtype)
                self.Pad_classifier.append(pickle.load(open(model_path, "rb")))
    def predict(self, input):
        if self.dataset_type == "timeonly":
            if len(input) != 2:
                raise Exception("Input shape does not match: {}, expect 2!".format(len(input)))
        elif self.dataset_type == "reduced":
            if len(input) != 6:
                raise Exception("Input shape does not match: {}, expect 6!".format(len(input)))
        elif self.dataset_type == "full":
            if len(input) != 12:
                raise Exception("Input shape does not match: {}, expect 12!".format(len(input)))
        elif self.dataset_type == "all":
            if len(input) != 3:
                raise Exception("Input shape (2D-list) does not match: {}, expect 3!".format(len(input)))
        else:
            raise Exception("Dataset type does not exist!")
        if self.dataset_type != "all":
            X_sample = np.array(input).reshape(1, -1)
            IC = self.IC_classifier.predict(X_sample)[0]
            OC = self.OC_classifier.predict(X_sample)[0]
            Kernel = self.Kernel_classifier.predict(X_sample)[0]
            Stride = self.Stride_classifier.predict(X_sample)[0]
            Pad = self.Pad_classifier.predict(X_sample)[0]
        else:
            IC_list = []
            OC_list = []
            Kernel_list = []
            Stride_list = []
            Pad_list = []
            X_sample = []
            X_sample.append(np.array(input[0]).reshape(1, -1))
            X_sample.append(np.array(input[1]).reshape(1, -1))
            X_sample.append(np.array(input[2]).reshape(1, -1))
            for i in range(3):
                IC_list.append(self.IC_classifier[i].predict(X_sample[i])[0])
                OC_list.append(self.OC_classifier[i].predict(X_sample[i])[0])
                Kernel_list.append(self.Kernel_classifier[i].predict(X_sample[i])[0])
                Stride_list.append(self.Stride_classifier[i].predict(X_sample[i])[0])
                Pad_list.append(self.Pad_classifier[i].predict(X_sample[i])[0])
            IC = sum(IC_list)/len(IC_list)
            OC = sum(OC_list)/len(OC_list)
            Kernel = sum(Kernel_list)/len(Kernel_list)
            Stride = sum(Stride_list)/len(Stride_list)
            Pad = sum(Pad_list)/len(Pad_list)
        Prediction = "Prediction is:\n Conv2D({}, {}, kernel_size = {}, stride = {}, padding = {})".format(IC, OC, Kernel, Stride, Pad)
        return Prediction

def main(args):
    model_path = "./saved_models/RF_{}_minsplit_{}_{}_TargetIC.pickle".format(args.n_estimators, args.min_samples_split, args.dataset_type)
    IC_classifier = pickle.load(open(model_path, "rb"))

    model_path = "./saved_models/RF_{}_minsplit_{}_{}_TargetOC.pickle".format(args.n_estimators, args.min_samples_split, args.dataset_type)
    OC_classifier = pickle.load(open(model_path, "rb"))

    model_path = "./saved_models/RF_{}_minsplit_{}_{}_TargetKernel.pickle".format(args.n_estimators, args.min_samples_split, args.dataset_type)
    Kernel_classifier = pickle.load(open(model_path, "rb"))

    model_path = "./saved_models/RF_{}_minsplit_{}_{}_TargetStride.pickle".format(args.n_estimators, args.min_samples_split, args.dataset_type)
    Stride_classifier = pickle.load(open(model_path, "rb"))

    model_path = "./saved_models/RF_{}_minsplit_{}_{}_TargetPad.pickle".format(args.n_estimators, args.min_samples_split, args.dataset_type)
    Pad_classifier = pickle.load(open(model_path, "rb"))

    X_sample = np.array([556216.0,13816.0,58.62,77.15,22199725.0,200704.0,89.71,25.04,5237356.0,162306.0,33842932.0,112]).reshape(1, -1)
    IC = IC_classifier.predict(X_sample)[0]
    OC = OC_classifier.predict(X_sample)[0]
    Kernel = Kernel_classifier.predict(X_sample)[0]
    Stride = Stride_classifier.predict(X_sample)[0]
    Pad = Pad_classifier.predict(X_sample)[0]

    print("Prediction for Input {} is:\n Conv2D({}, {}, kernel_size = {}, stride = {}, padding = {})".format(str(X_sample), IC, OC, Kernel, Stride, Pad))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="timeonly", help='Pick dataset you want to generate', choices=("reduced", "full", "timeonly"))
    parser.add_argument("--n_estimators", type=int, default=50, help='Number of Trees for Random Forest')
    parser.add_argument("--min_samples_split", type=int, default=30, help='Minimum Number of Splitting Features for Each Split')
    args = parser.parse_args()
    main(args)
