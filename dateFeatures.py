import os
import pandas as pd
import datefinder
from datetime import datetime
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import average_precision_score
import re

path = "./Downloads/Timeline17/Timeline17/Data/"
topics = os.listdir(path)
features = []
frames = []
labels = []
framestest = []
labelstest = []
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
for f in topics:
	if os.path.isdir(os.path.join(path,f)):
		dates = os.listdir(os.path.join(path,f+"/InputDocs"))
		dates = [da for da in dates if os.path.isdir(os.path.join(path,f+"/InputDocs/"+da))]
		dates.sort()
		c = ['Is Date Selected']
		rows = [tdate +'_'+ f for tdate in dates]
		isDateSelected = pd.DataFrame(0,index = rows, columns = c)
		

		timeline = os.listdir(os.path.join(path,f+"/timelines"))
		fname = os.path.join(path,f+"/timelines/"+timeline[0])
		ipfile = open(fname, 'r') 
		text = ipfile.readlines()
		for t in text:
			if re.match(r'\d{4}-\d{2}-\d{2}$',t):
				if t[0:10] in dates:
					isDateSelected.loc[t[0:10]+'_'+f,'Is Date Selected'] = 1

		#if f=='IraqWar_guardian':
		#labelstest.append(isDateSelected)
		#else:
		labels.append(isDateSelected)		


		articleCount = pd.DataFrame(0,index = dates, columns = dates)
		sentenceCount = pd.DataFrame(0,index = dates, columns = dates)
		for d in dates:
			if os.path.isdir(os.path.join(path,f+"/InputDocs/"+d)):			
				articles = os.listdir(os.path.join(path,f+"/InputDocs/"+d))
				for a in articles:			
					fname = os.path.join(path,f+"/InputDocs/"+d+"/"+a)
					ipfile = open(fname, 'r') 
					text = ipfile.read()
					matches = datefinder.find_dates(text,strict='true')
					unique = []
					for m in matches:
						if m.year>2014:
							y=datetime.strptime(d,'%Y-%m-%d').year
							m=m.replace(year=y)
						mdate = m.date().isoformat()
						try:
							sentenceCount.loc[d,mdate] = sentenceCount.loc[d,mdate]+1	
						except:
							pass
						if m.date() not in unique:
							unique.append(m.date())
					for u in unique:
						udate=u.isoformat()
						try:
							articleCount.loc[d,udate] = articleCount.loc[d,udate]+1	
						except:
							pass
		#print articleCount

		columns = ['Articles On date','Articles After Date','Articles Before Date','Sentences On Date','Sentences After Date','Sentences Before Date']
		df = pd.DataFrame(index = rows,columns = columns)		
		for date in dates:
			if os.path.isdir(os.path.join(path,f+"/InputDocs/"+date)):
				s = date+'_'+f
				df.loc[s,'Articles On date'] = len(os.listdir(os.path.join(path,f+"/InputDocs/"+date)))
				df.loc[s,'Articles After Date'] = (articleCount.loc[date:,date].sum())-articleCount.loc[date,date]
				df.loc[s,'Articles Before Date'] = (articleCount.loc[:date,date].sum())-articleCount.loc[date,date]
				df.loc[s,'Sentences On Date'] = sentenceCount.loc[date,date]
				df.loc[s,'Sentences After Date'] = (sentenceCount.loc[date:,date].sum())-sentenceCount.loc[date,date]
				df.loc[s,'Sentences Before Date'] = (sentenceCount.loc[:date,date].sum())-sentenceCount.loc[date,date]
		#print 'Topic: ',f
		#print df
		#if f=='IraqWar_guardian':
		#framestest.append(df)
		#else:
		frames.append(df)		


allfeaturesdf = pd.concat(frames)
alllabelsdf = pd.concat(labels)
#featurestestdf = pd.concat(framestest)	
#labelstestdf = pd.concat(labelstest)	


allfeaturesdf.to_csv('DateFeatures.csv')
alllabelsdf.to_csv('DateLabels.csv')
