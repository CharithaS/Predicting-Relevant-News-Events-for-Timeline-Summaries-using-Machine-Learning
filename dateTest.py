import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import average_precision_score
path = "./Downloads/Timeline17/Timeline17/Data/"
topics = os.listdir(path)
allfeaturesdf = pd.read_csv('DateFeatures.csv',index_col=0)
alllabelsdf = pd.read_csv('DateLabels.csv',index_col=0)
for top in topics:
	if os.path.isdir(os.path.join(path,top)):
		labelsdf = alllabelsdf[[top not in x for x in alllabelsdf.index]]
		featuresdf = allfeaturesdf[[top not in x for x in allfeaturesdf.index]]
		labelstestdf = alllabelsdf[[top in x for x in alllabelsdf.index]]
		featurestestdf = allfeaturesdf[[top in x for x in allfeaturesdf.index]]
		clf = SVC()
		clf.fit(featuresdf.values,labelsdf.values)
		output = clf.predict(featurestestdf.values)
		acc = clf.score(featurestestdf.values,labelstestdf.values)
		print 'Topic:',top
		print 'Accuracy'
		print acc
