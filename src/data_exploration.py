import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree
import config
import matplotlib.pyplot as plt
df_train = pd.read_csv(config.TRAINING_FILE)
X_train = df_train.drop('label',axis =1).values
y_train = df_train.label.values
#print(y_train[0])

#print(X_train[0])
single_image = X_train[4].reshape(28,28)
plt.imshow(single_image,cmap ='gray')
plt.show()