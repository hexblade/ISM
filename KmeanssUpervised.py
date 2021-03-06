package = "sklearn.datasets"
name1 = input("what dataset would you like to import? ")
name = "load_"+name1
datafn = getattr(__import__(package, fromlist=[name]), name)
iris = datafn()
x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)

from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)

from sklearn.metrics import accuracy_score
numacc = (accuracy_score(y_test,predictions)) * 100
goodness = True
if numacc < 95:
  goodness = False
stracc = str(numacc)
straccpercent = stracc[:5] + '%'
if goodness == True:
  print ("Nice! the accuracy of the model was", straccpercent)
if goodness == False:
  print ("Oof... the accuracy of the model was", straccpercent)
