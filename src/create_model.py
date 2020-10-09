import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree
import config
from sklearn import model_selection
import argparse
import model_dispatcher
import os

def run(model):
    #training dataset
    df_train = pd.read_csv(config.TRAINING_FILE)
    df_train = df_train.sample(frac = 1,random_state= 42) 
    X_train = df_train.drop('label',axis =1).values
    y_train = df_train.label.values
    #validation dataset
    df_test = pd.read_csv(config.TESTING_FILE)
    X_test = df_test.drop('label',axis =1).values
    y_test = df_test.label.values
    #use decision tree classifier
    clf = model_dispatcher.models[model]
    #fit the model
    clf.fit(X_train,y_train)
    #predicts for validation sample
    pred = clf.predict(X_test)
    #calculate accuracy
    accuracy = metrics.accuracy_score(y_test,pred)
    print('accuracy of the model is :', accuracy)
    #save the model
    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT,f"{model}.bin"))

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str
    )
args = parser.parse_args()

run(model=args.model)