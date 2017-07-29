import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing, cross_validation, svm

features = ['BpoilFoxnews','BpoilGuardian','BpoilReuters','BpoilWashington','H1N1BBC','H1N1Guardian','H1N1Reuters','HaitiBBC','IraqWarGuardian','LibyaWarCNN','LibyaWarReuters','MJBBC','SyrianCrisisBBC','SyrianCrisis_reuters']

sentFeaturesdf = pd.read_csv('BpoilBBCFeatures.csv',index_col=0)
sentLabelsdf = pd.read_csv('BpoilBBCLabels.csv',index_col=0)
for feat in features:
	dfFeat = pd.read_csv(feat+'Features.csv',index_col=0)
	sentFeaturesdf = pd.concat([sentFeaturesdf,dfFeat])
	dfLabel = pd.read_csv(feat+'Labels.csv',index_col=0)
	sentLabelsdf = pd.concat([sentLabelsdf,dfLabel])

for z in sentLabelsdf.index:
	sentLabelsdf.loc[z,'Max Cosine Similarity']=sentLabelsdf.loc[z,'Max Cosine Similarity'][2:-1]

print sentLabelsdf

labelsdf = sentLabelsdf[['SyrianCrisis_bbc' not in x for x in sentLabelsdf.index]]
print 'Train Labels'
print labelsdf
featuresdf = sentFeaturesdf[['SyrianCrisis_bbc' not in x for x in sentFeaturesdf.index]]
print 'Train Features'
print featuresdf
labelstestdf = sentLabelsdf[['SyrianCrisis_bbc' in x for x in sentLabelsdf.index]]
print 'Test Labels'
print labelstestdf
featurestestdf = sentFeaturesdf[['SyrianCrisis_bbc' in x for x in sentFeaturesdf.index]]
print 'Test Features'
print featurestestdf

clf = linear_model.LinearRegression()
clf.fit(featuresdf.values,labelsdf.values)
output = clf.predict(featurestestdf.values)
print 'Output'
print output
meanAbsError = mean_absolute_error(labelstestdf.values,output)
print 'Mean Absolute Error'
print meanAbsError
meanSquareError = mean_squared_error(labelstestdf.values,output)
print 'Mean Square Error'
print meanSquareError
score = clf.score(featurestestdf.values,labelstestdf.values)
print 'Score'
print score

print 'SVR'
clfSVR = svm.SVR()
clfSVR.fit(featuresdf.values,labelsdf.values)
outputSVR = clfSVR.predict(featurestestdf.values)
print 'Output'
print outputSVR
meanAbsErrorSVR = mean_absolute_error(labelstestdf.values,outputSVR)
print 'Mean Absolute Error'
print meanAbsErrorSVR
meanSquareErrorSVR = mean_squared_error(labelstestdf.values,outputSVR)
print 'Mean Square Error'
print meanSquareErrorSVR
scoreSVR = clfSVR.score(featurestestdf.values,labelstestdf.values)
print 'Score'
print scoreSVR
