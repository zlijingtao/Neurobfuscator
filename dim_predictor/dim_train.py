import argparse
from features import SimpleTransform, FeatureMapper, MaskedPCA
from sklearn.preprocessing import FunctionTransformer
from sklearn import  preprocessing
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.decomposition import PCA

def get_pipeline(n_estimators = 50, min_samples_split = 10):
    steps = [("scaler", StandardScaler()),
             ("classify", RandomForestRegressor(n_estimators=n_estimators,
                                                verbose=2,
                                                n_jobs=4,
                                                min_samples_split=min_samples_split,
                                                random_state=1234))]
    return Pipeline(steps)

def main(args):
    train_path = "../seq_predictor/obfuscator/dataset/{}_{}_classification.csv".format(args.dataset_type, args.operator_name)
    model_path = "./saved_models/{}_RF_{}_minsplit_{}_{}_{}.pickle".format(args.operator_name, args.n_estimators, args.min_samples_split, args.dataset_type, args.target)

    print("Reading in the training data")
    train = pd.read_csv(train_path)
    

    '''Separate Feature and Label'''
    try:
        y = train[args.target]
    except:
        raise("Target does not exist!")
    
    if args.operator_name == "conv":
        try:
            X = train.drop(columns=['TargetIC', 'TargetOC', 'TargetKernel', 'TargetStride', 'TargetPad'])
            # X = X.drop(columns=['FeatureImageDim'])
        except:
            raise("Some Target do not exist, please check the integrity of training file!")
    elif args.operator_name == "fc":
        try:
            X = train.drop(columns=['TargetDim'])
            # X = X.drop(columns=['FeatureDim'])
        except:
            raise("Some Target do not exist, please check the integrity of training file!")
    elif args.operator_name == "depth":
        try:
            X = train.drop(columns=['TargetIC', 'TargetOC'])
            # X = X.drop(columns=['FeatureImageDim'])
        except:
            raise("Some Target do not exist, please check the integrity of training file!")

    '''Train Test Split'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    print("Processing the raw features")
    print(X_train.head(5))
    classifier = get_pipeline(args.n_estimators, args.min_samples_split)

    '''WARNING!!!'''
    option = args.option
    '''(Option-0: Train) Use all training data to train the model'''
    if option == 0:
        #Cost about 5 min
        start_time = time.time()
        classifier.fit(X, y)
        time_cost = time.time() - start_time
        print("Time cost is", time_cost, "s")
        print("Training is done, please run the predict.py and submit.")
    elif option == 1:
        '''(Option-1: Standard Train/Validate) Use hold-out validation set to check the mae_score'''
        #Cost about 5 min
        start_time = time.time()

        classifier.fit(X_train, y_train)
        time_cost = time.time() - start_time
        mae_score_train = mean_absolute_error(y_train, classifier.predict(X_train))
        mae_score_test = mean_absolute_error(y_test, classifier.predict(X_test))
        print("Time cost is", time_cost)
        print("Mean Absolute Error on the Traning set is: %.4f" %mae_score_train)
        print("Mean Absolute Error on the validation set is: %.4f" %mae_score_test)
    elif option == 2:
        '''(Option-2: Grid Search for best hyperparameter) Uncomment below to do grid search'''
        #Cost about 2 hour
        scoring = {'mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better = False)}
        param_grid = {'classify__n_estimators': [60, 80, 100, 120, 140]}
        search = GridSearchCV(classifier, param_grid, scoring = scoring, cv=5, refit = 'mean_absolute_error').fit(X, y)
        print(search.best_params_)
        with open('./saved_models/{}_RF_{}_minsplit_{}_{}_{}_grid_search.pickle'.format(args.operator_name, args.n_estimators, args.min_samples_split, args.dataset_type, args.target), 'wb') as handle:
            pickle.dump(search, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        '''Option-3: Load existing model'''
        classifier = pickle.load(open(model_path, "rb"))
        predictions = classifier.predict(X_test)
        mae_score_test = mean_absolute_error(y_test, predictions)

        print("DER for current dimension parameter is: ", np.mean(np.abs(y_test - predictions)/predictions))
        print("Mean Absolute Error on the validation set is: %.4f" %mae_score_test)

    print("Saving the classifier")
    with open(model_path, 'wb') as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="timeonly", help='Pick dataset you want to generate', choices=("reduced", "full", "timeonly"))
    parser.add_argument("--target", type=str, default="TargetIC", choices = ['TargetIC', 'TargetOC', 'TargetKernel', 'TargetStride', 'TargetPad', 'TargetDim'], help='which dim to predict')
    parser.add_argument("--operator_name", type=str, default="conv", choices = ['conv', 'fc', 'depth'], help='which operator to predict')
    parser.add_argument("--option", type=int, default=1, choices = [0, 1, 2, 3], help='Training Options: 0: train with 100& data, 1: standard train/test, 2: grid search for hyper-params, 3: load and test')
    parser.add_argument("--n_estimators", type=int, default=50, help='Number of Trees for Random Forest')
    parser.add_argument("--min_samples_split", type=int, default=30, help='Minimum Number of Splitting Features for Each Split')
    args = parser.parse_args()
    main(args)
