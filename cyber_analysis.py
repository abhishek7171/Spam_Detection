import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,MultinomialNB
import graphviz
import io

#using enter dataset and then manipulating
df = pd.read_csv("real_cyber.csv", header=None, error_bad_lines=False, index_col=False, dtype='unicode')
X_1 = pd.read_csv("fullcyber_data.csv", header=None, error_bad_lines=False, index_col=False, dtype='unicode')

#displaying the head of the dataset
df.head()
df.columns = [str(x) for x in df.columns]

#displaying information of all the features present within the dataset
df.info()

#replacing output string labels with integer labels for classification
df['12'] = df['12'].replace(['spammer' , 'non-spammer'], [1,0])
df.head()

#predicted labels stored in y
y = df["12"]

#data to be trained on, stored as X
X = df.drop(['12','9'], axis=1, inplace=True)
#new data frame 
df.head()
df = df.drop(df.index[0]) #dropping string datatypes for fitting.

#y=df["12"]
#X_1

#splitting the data as training data and test data
X_train,X_test,y_train,y_test = train_test_split(df.as_matrix(),y,test_size=0.25,random_state=5)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

#USING ALL MODELS WITH DEFAULT PARAMETERS. MODEL CAN BE TWEAKED DEPENDING ON DATASET.

#calling the decision tree classifier on the training dataset.
clf = tree.DecisionTreeClassifier(criterion='gini')
y_1 = np.asarray(y_train) #converting the predicted trained data to 1D array for fitting.
y_1

clf = preprocessing.OneHotEncoder() #one hot encoding, only when required

clf.fit(X_train,y_1) #fitting the model for training.

#df.info()
#y_train = np.asarray(df['1'],dtype="|S6")
X_train
#displaying mean score on train data itself.
print(clf.score(X_train,y_train))

#displaying mean score with respect to predicted output.
clf.score(X_test,y_test)
X_test

#predicting the labels as spam or non-spam for this model
predicted = clf.predict(X_test)
predicted	

#Generating the tree for visualization of the dataset 
with open("spam_classifier.txt", "w") as f:
    f = tree.export_graphviz(clf, out_file=f)

#Logistic regression for classification problem.
#using regularization parameter of 1e5.
clf_1 = linear_model.LogisticRegression(C=1e5)

#fitting data into the logistic regression model.
clf_1.fit(X_train,y_train)
print(clf_1.score(X_train,y_train))
clf_1.score(X_test,y_test)#displaying mean score for Logistic Regression with respect to predicted output.

#predicting the labels as spam or non-spam for this model
predicted = clf_1.predict(X_test)
predicted	

#Random Forest model.
clf_2 = RandomForestClassifier(#maxdepth = (depending on data)
			,bootstrap=True, class_weight=None, criterion='gini',
            max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
#Fitting the data into the RandomForestClassifier.
clf_2.fit(X_train,y_train)
#Predicting the mean score for Random forest with respect to predicted output.
clf_2.score(X_train,y_train)
clf_2.score(X_test,y_test)
#predicting the labels as spam or non-spam for this model
predicted = clf_2.predict(X_test)
predicted

#SVM (Linear SVC)
#Taking mean of the predicted score over 10 iterations to generalise output predicted accuracy.
sum=0
i=1
while(i<11):
    clf_3 = svm.LinearSVC(C=1) #calling Svm classifier with C=1.
    clf_3.fit(X_train,y_train)  # fitting the data into the model.
    print(clf_3.score(X_train,y_train))
    sum=sum+clf_3.score(X_test,y_test)  #Accuracy for one iteration.
    i=i+1
sum=sum/10
print(sum) #mean score 
#predicting the labels as spam or non-spam for this model
predicted = clf_3.predict(X_test)
predicted

#NaiveBayes Classifier, using the gaussian NB classifier.
clf_4 = GaussianNB(priors = None)
clf_4.fit(X_train,y_train)#fitting the data into the classifier.
clf_4.score(X_train,y_train)
clf_4.score(X_test,y_test) #predicting the score for this model.
#predicting the labels as spam or non-spam for this model
predicted = clf_4.predict(X_test)
predicted


    
