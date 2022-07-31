import pickle
import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\user\\Desktop\\Deep Learning\\13 day -django\\DeployModel_DJango\\Deploymodel\\iris.data')

X = np.array(df.iloc[:, 0:4])
Y = np.array(df.iloc[:, 4:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test ,Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,Y_train)


pickle.dump(sv, open('iri.pkl', 'wb'))