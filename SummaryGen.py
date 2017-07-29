import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import average_precision_score
import re
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
path = "./Downloads/Timeline17/Timeline17/Data/"
topics = os.listdir(path)
allfeaturesdf = pd.read_csv('DateFeatures.csv',index_col=0)
alllabelsdf = pd.read_csv('DateLabels.csv',index_col=0)

labelsdf = alllabelsdf[['SyrianCrisis_bbc' not in x for x in alllabelsdf.index]]
featuresdf = allfeaturesdf[['SyrianCrisis_bbc' not in x for x in allfeaturesdf.index]]
labelstestdf = alllabelsdf[['SyrianCrisis_bbc' in x for x in alllabelsdf.index]]
featurestestdf = allfeaturesdf[['SyrianCrisis_bbc' in x for x in allfeaturesdf.index]]

features = ['BpoilFoxnews','BpoilGuardian','BpoilReuters','BpoilWashington','H1N1BBC','H1N1Guardian','H1N1Reuters','HaitiBBC','IraqWarGuardian','LibyaWarCNN','LibyaWarReuters','MJBBC','EgyptianProtest_cnn','SyrianCrisis_reuters']

sentFeaturesdf = pd.read_csv('./TextSummarisation/BpoilBBCFeatures.csv',index_col=0)
sentLabelsdf = pd.read_csv('./TextSummarisation/BpoilBBCLabels.csv',index_col=0)
for feat in features:
	dfFeat = pd.read_csv('./TextSummarisation/'+feat+'Features.csv',index_col=0)
	sentFeaturesdf = pd.concat([sentFeaturesdf,dfFeat])
	dfLabel = pd.read_csv('./TextSummarisation/'+feat+'Labels.csv',index_col=0)
	sentLabelsdf = pd.concat([sentLabelsdf,dfLabel])

for z in sentLabelsdf.index:
	sentLabelsdf.loc[z,'Max Cosine Similarity']=sentLabelsdf.loc[z,'Max Cosine Similarity'][2:-1]

clf = linear_model.LinearRegression()
clf.fit(sentFeaturesdf.values,sentLabelsdf.values)

syrianCrisisSentFeaturesdf = pd.read_csv('./TextSummarisation/SyrianCrisisBBCFeatures.csv',index_col=0)

file = open("Summaryfile.txt","w") 
 
i=1
#print syrianCrisisSentFeaturesdf
for index,row in labelstestdf.iterrows():
	if row['Is Date Selected']==1:
		date = re.search(r'.*(\d{4}-\d{2}-\d{2}).*',index,re.DOTALL).group(1)
		file.write(date)
		file.write("\n") 
		article = os.listdir("./Downloads/Timeline17/Timeline17/Data/SyrianCrisis_bbc/InputDocs/"+date)
		article.sort()
		textFromArt = []
		for arti in article:		
			fname = "./Downloads/Timeline17/Timeline17/Data/SyrianCrisis_bbc/InputDocs/"+date+"/"+arti
			art = open(fname,'r')
			artText = art.readlines ()
			textFromArt = textFromArt+artText
		print 'Date Text'
		print textFromArt
		syrianCrisisSenFeaturesDF = syrianCrisisSentFeaturesdf[[date in x for x in syrianCrisisSentFeaturesdf.index]]
		print 'Date'
		print date
		output = clf.predict(syrianCrisisSenFeaturesDF.values)
		predictedValuessdf = pd.DataFrame(data=output,index=syrianCrisisSenFeaturesDF.index.values,columns=['Pred Value'])
		predictedValuessdf = predictedValuessdf.sort_values(by='Pred Value',ascending=0)
		topFourSen = predictedValuessdf[0:4].index.values
		print 'top 4'
		print topFourSen
		file1 = open("./SysSummary/DateSummary"+str(i)+".txt","w")
		for sen in topFourSen:
			senDetails = re.search(r'.*_(.*)_(.*)$',sen,re.DOTALL)
			fileName = senDetails.group(1)
			position = senDetails.group(2)
			art = open("./Downloads/Timeline17/Timeline17/Data/SyrianCrisis_bbc/InputDocs/"+date+"/"+fileName,'r')
			artText = art.readlines ()
			file.write(textFromArt[int(position)]) 
			file1.write(textFromArt[int(position)])
		i=i+1


		
