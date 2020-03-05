from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

package = "sklearn.datasets"
name1 = input("what dataset would you like to import? ")
name = "load_"+name1
getattr(__import__(package, fromlist=[name]), name)

from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd, numpy as np

class Iris(object):
    
    def __init__(self):
        self.load_data()
        return None
    
    def load_data(self):
        package = "sklearn.datasets"
        name1 = input("what dataset would you like to import? ")
        name = "load_"+name1
        datafn = getattr(__import__(package, fromlist=[name]), name)
        data = datafn()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
        return 1
    
    def Kmeans(self, technique='random', n_clusters=2, output = 'all'): 
        km = KMeans(init=technique, n_clusters=n_clusters)    
        km.fit(self.X_train)
        self.X_train = pd.DataFrame(self.X_train)
        self.X_test = pd.DataFrame(self.X_test)
        if output == 'all':
            self.X_train['km'] = km.labels_
            self.X_test['km'] = km.predict(self.X_test)
        elif output == 'one':
            self.X_train = km.labels_.reshape(-1, 1)
            self.X_test = km.predict(self.X_test).reshape(-1, 1) 
        return self
    
    def model(self, model = LogisticRegression()):
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)

#numacc = Iris().model() * 100

numacc = Iris().Kmeans(n_clusters = 3, output = 'one').model() * 100

#numacc = Iris().Kmeans(n_clusters = 3, output = '3').model() * 100

goodness = True
if numacc < 95:
  goodness = False
stracc = str(numacc)
straccpercent = stracc[:5] + '%'
if goodness == True:
  print ("Nice! the accuracy of the model was", straccpercent)
if goodness == False:
  print ("Oof... the accuracy of the model was", straccpercent)
