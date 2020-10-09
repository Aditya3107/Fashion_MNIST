# src/train.py 
import pandas as pd 
import config
from sklearn import model_selection

if __name__=='__main__':
    df = pd.read_csv(config.TRAINING_FILE)
    df['kfold'] = -1
    df = df.sample(frac = 1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=5)
    for fold,(t_,v_) in enumerate(kf.split(X=df)):
        df.loc[v_,'kfold'] = fold
    df.to_csv('input/fashion_mnist_kfold.csv',index = False)

