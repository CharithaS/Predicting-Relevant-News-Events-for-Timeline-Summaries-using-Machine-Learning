import codecs
from spacy.en import English
import os
import pandas as pd
import datefinder
from datetime import datetime
import numpy as np
import re
from nltk_contrib.timex import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import string
import math
#nlp = spacy.load("en")
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing, cross_validation, svm
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
ps = PorterStemmer()
path = "./Downloads/Timeline17/Timeline17/Data/"
topics = os.listdir(path)
sentFrames =[]
sentLabelFrames =[]
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
causalConnectives = ['so', 'because', 'in case', 'thus', 'in order', 'except', 'as', 'an effect of', 'though', 'moreover', 'still', 'yet', 'if', 'resulting in', 'nonetheless', 'since', 'stemmed from', 'by', 'accordingly', 'therefore', 'an upshot of', 'in that case', 'until', 'apart from', 'then', 'unless', 'under the circumstances', 'however', 'but', 'in order to', 'for that reason', 'although', 'despite', 'in conclusion', 'nevertheless', 'admittedly', 'all the same', 'to that end', 'due to', 'hence', 'an outcome of', 'consequently', 'in this way', 'otherwise','thanks to this','this causes','the reason that','this results in']

temporalConnectives = ['secondly', 'already', 'afterwards', 'just', 'eventually', 'previously', 'at the same time', 'soon', 'hitherto', 'second', 'eleventh', 'in due course', 'lastly', 'fifth', 'before', 'on another occasion', 'still', 'since', 'when', 'next', 'thirdly', 'finally', 'now that', 'fourth', 'as early as', 'until', 'then', 'recently', 'meanwhile', 'sixth', 'after', 'here', 'in the beginning', 'first of all', 'at this point', 'now', 'straightaway', 'in the end', 'last', 'third', 'whenever', 'at this moment', 'in the mean time', 'later', 'while', 'firstly', 'subsequently', 'once', 'after some time', 'first']

logicalConnectives = ['in addition', 'and', 'it is false that', 'indeed', 'anyway', 'after all', 'moreover', 'if', 'even', 'this is an important issue because', 'similarly', 'furthermore', 'also', 'too', 'just in case', 'neither', 'it is not the case that', 'a further point', 'however', 'but', 'many people believe', 'besides', 'not', 'like', 'not both', 'implies', 'claim that', 'let alone', 'one reason is', 'that is to say', 'or']

for f in topics:
	print 'Topic'+f
	textFromArt = []
	textForDate = {}
	if os.path.isdir(os.path.join(path,f)):
		timeline = os.listdir(os.path.join(path,f+"/timelines"))
		fname = os.path.join(path,f+"/timelines/"+timeline[0])
		ipfile = open(fname, 'r') 
		tsText = ipfile.read()
		dates = os.listdir(os.path.join(path,f+"/InputDocs"))
		dates = [da for da in dates if os.path.isdir(os.path.join(path,f+"/InputDocs/"+da))]
		dates.sort()
		for dat in dates:
			textDate =''
			if os.path.isdir(os.path.join(path,f+"/InputDocs/"+dat)):
				isDateinSum = re.search(dat+r'(.+?)-{32}',tsText,re.DOTALL)
				if isDateinSum:
					article = os.listdir(os.path.join(path,f+"/InputDocs/"+dat))
					article.sort()
					for arti in article:		
						fname = os.path.join(path,f+"/InputDocs/"+dat+"/"+arti)
						art = open(fname,'r')
						artText = art.read ()
						textFromArt.append(artText)
						textDate = textDate + artText
					textForDate[dat] = textDate

		cv = CountVectorizer(analyzer='word', stop_words='english')
		cv_fit=cv.fit_transform(textFromArt)
		featureWords = cv.get_feature_names()
		tfTopicdf = pd.DataFrame(data=cv_fit.toarray(),columns=featureWords)
		colToDrop=[]
		for col in tfTopicdf.columns.values:
			if re.match(r'[0-9]+',col):
				colToDrop.append(col)

		tfTopicdf = tfTopicdf.drop(colToDrop, axis=1)
		#print tfTopicdf
		tfidf = TfidfVectorizer (analyzer='word', stop_words='english')
		tfs = tfidf.fit_transform (textFromArt)
		tfs = tfs.todense()
		feature_names = tfidf.get_feature_names ()
		i = 0
		for d in dates:
			if os.path.isdir(os.path.join(path,f+"/InputDocs/"+d)):	
				mt = re.search(d+r'(.+?)-{32}',tsText,re.DOTALL)
				if mt:
					tsDate = mt.group(1)				
					articles = os.listdir(os.path.join(path,f+"/InputDocs/"+d))
					articles.sort()
					columns = ['Length','Stop Words','Non Stop Words','Pronouns','Position','Causal Signals','Temporal Signals','Temporal Expressions','Logical Signals','Sum TFIDF','Avg TFIDF','pos * TFIDF','TF Top10','TF Top30','TF Top50','TF Top100','Sum LogOdd','Top LogOdd','Avg LogOdd','Cross Entropy','Has Temporal Expr']
					df = pd.DataFrame(columns = columns)
					index=0
					labelcolumns = ['Max Cosine Similarity']
					sentLabels = pd.DataFrame(columns = labelcolumns)
					for a in articles:			
						fname = os.path.join(path,f+"/InputDocs/"+d+"/"+a)
						docText = open(fname).read()
						sentFromTsandArt = tsDate.splitlines() + docText.splitlines()
						for sind in range(len(sentFromTsandArt)):
							words = word_tokenize(sentFromTsandArt[sind])
							for wrd in words:
								sentFromTsandArt[sind] = sentFromTsandArt[sind].replace(wrd,ps.stem(wrd))
						cv2 = CountVectorizer(analyzer='word', stop_words='english')
						cv_fit2 = cv2.fit_transform(sentFromTsandArt)
						termFreq = cv_fit2.toarray()
						termFreq[termFreq>1]=1
						rows=len(termFreq)
						cols=len(termFreq[0])
						print 'Date nd Article:',d+a
						print 'Rows',rows
						print 'Columns',cols

						tfidfSen = np.empty((rows, cols), dtype=object)

						for r in range(rows):
							for c in range(cols):
								docFreq = sum(row[c] for row in termFreq)
								#print termFreq[r,c]
								tfidfSen[r,c] = termFreq[r,c] * math.log10(float(rows-1)/float(docFreq))
						#print 'tfidfSen'
						#print tfidfSen
						tsLength = len(tsDate.splitlines())
						sentPosition = 0
						tfCount = cv_fit.toarray()[i]
						tfCountdf = pd.DataFrame(data=tfCount,index=cv.get_feature_names(),columns=['Count'])
						tfCountdf = tfCountdf.sort_values(by='Count', ascending=0)
						digits=[]
						for ind,row in tfCountdf.iterrows():
							if re.match(r'[0-9]+',ind):
								digits.append(ind)
						tfCountdf = tfCountdf.drop(digits)
						topTen = []
						topThirty = []
						topFifty = []
						topHundred = []
						if len(tfCountdf.index.values) >= 10:
							topTen = tfCountdf[0:10].index.values
						if len(tfCountdf.index.values) >= 30:
							topThirty = tfCountdf[0:30].index.values
						if len(tfCountdf.index.values) >= 50:
							topFifty = tfCountdf[0:50].index.values
						if len(tfCountdf.index.values) >= 100:
							topHundred = tfCountdf[0:100].index.values
						for sen in docText.splitlines():
							parser =  English ()
							parsedData = parser(unicode(sen))
							crossEntropy = 0
							sumLogOdd = 0
							noOfWords = 0
							maxLogOdd = -9999
							avgLogOdd = 0
							topTenCount = 0
							topThirtyCount = 0
							topFiftyCount = 0
							topHundredCount = 0
							causalCount = 0
							temporalCount = 0 
							logicalCount = 0
							print 'Sentence'
							print sen
							print 'Index+length'
							print sentPosition+tsLength
							cos = cosine_similarity(tfidfSen[sentPosition+tsLength].reshape(1,-1), tfidfSen[1:tsLength])
							#print 'Cosine Similarity'
							#print cos
							maxCosine = cos.max(axis = 1)
							#print 'MaxCosine'
							#print maxCosine
							inde = f+'_'+d+'_'+a+'_'+str(index)
							sentLabels.loc[inde,'Max Cosine Similarity'] = maxCosine 
							for ca in causalConnectives:
		 						reg=r'\b'+ca+r'\b'
								if re.search(reg,sen.lower()):
									causalCount = causalCount + 1
							for te in temporalConnectives:
		 						reg=r'\b'+te+r'\b'
								if re.search(reg,sen.lower()):
									temporalCount = temporalCount + 1
							for lo in logicalConnectives:
		 						reg=r'\b'+lo+r'\b'
								if re.search(reg,sen.lower()):
									logicalCount = logicalCount + 1
							taggedSen = tag(sen)
							hasTempExpr = 0
							if len(list(re.finditer(r'<TIMEX2>',taggedSen))) != 0:
								hasTempExpr = 1
							stopCount = 0
							nonStopCount = 0
							pronounCount = 0
							tfidfSum = 0
							no = 0
							tfidfpos = 0
							for x,token in enumerate(parsedData):
								if not token.is_punct and not token.is_stop and not re.match(r'[0-9]+',token.string) and token.text!=unicode("\n") and token.text!=unicode("'s") and token.text!=unicode('$'):
									tok = r'\b'+re.escape(token.text)+r'\b'
									countTokeninDate = sum(1 for count in re.finditer(tok,textForDate[d]))
									countTokeninSent = sum(1 for countSen in re.finditer(tok,sen))
									if countTokeninDate!=0 and countTokeninSent!=0:
										probTokeninDate = float(countTokeninDate)/float(len(textForDate[d]))
										probTokeninSent = float(countTokeninSent)/float(len(sen))
										crossEntropy = crossEntropy - (probTokeninDate * (math.log10(probTokeninSent)))
								if token.text.lower () in tfTopicdf.columns.values:
									print 'Token',token
									print i
									wordCountDoc = tfTopicdf.loc[i,token.text.lower()]
									wordCountOtherDoc = tfTopicdf.loc[[j for j in tfTopicdf.index.values if j!= i],token.text.lower ()].sum()
									otherWordsCountDoc = tfTopicdf.loc[i].sum() - wordCountDoc
									otherWordsCountOtherDoc = tfTopicdf.loc[[j for j in tfTopicdf.index.values if j!= i]].sum().sum() - wordCountOtherDoc
									logOdd = 0
									print 'wordCountDoc',wordCountDoc
									print 'wordCountOtherDoc',wordCountOtherDoc
									print 'otherWordsCountDoc',otherWordsCountDoc
									print 'otherWordsCountOtherDoc',otherWordsCountOtherDoc
									if wordCountOtherDoc != 0:
										ratio = (float(wordCountDoc) * float(otherWordsCountOtherDoc)) / (float(otherWordsCountDoc) * float(wordCountOtherDoc))
										print 'ratio',ratio
										logOdd = math.log10(ratio)
									sumLogOdd = sumLogOdd + logOdd
									noOfWords = noOfWords + 1
									if logOdd > maxLogOdd:
										maxLogOdd = logOdd

								if len(topTen) != 0:
									if token.text.lower () in topTen:
										topTenCount = topTenCount + 1
								if len(topThirty) != 0:
									if token.text.lower () in topThirty:
										topThirtyCount = topThirtyCount + 1
								if len(topFifty) != 0:
									if token.text.lower () in topFifty:
										topFiftyCount = topFiftyCount + 1
								if len(topHundred) != 0:
									if token.text.lower () in topHundred:
										topHundredCount = topHundredCount + 1 
								if token.text.lower () in feature_names:
									tfidfSum = tfidfSum + tfs[i,feature_names.index(token.text.lower ())]
									no = no + 1
									if token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'VERB':
										posWeight = 5
									elif token.pos_ == 'ADJ' or token.pos_ == 'ADV':
										posWeight = 3
									else:
										posWeight = 1
									tfidfpos = tfidfpos + (tfs[i,feature_names.index(token.text.lower ())]*posWeight)
								if token.is_stop:
									stopCount =stopCount + 1
									#print 'Stop'
									#print token
								elif not token.is_punct and token.text!=unicode("\n") and token.text!=unicode("'s"):
									nonStopCount = nonStopCount + 1
									#print 'Non Stop'
									#print token
								if token.pos_ == unicode("PRON"):
									#print "Pronoun"
									#print token
									pronounCount = pronounCount + 1
							if noOfWords != 0:
								avgLogOdd = sumLogOdd/float(noOfWords)
							avg = tfidfSum/float(no)	
							length = stopCount + nonStopCount
							sentPosition = sentPosition + 1
							#print sen 
							#print 'Length:'
							#print length
							#print 'Stop:'
							#print stopCount
							#print 'Non Stop:'
							#print nonStopCount
							#print 'Pronoun:'
							#print pronounCount
							#print 'Sentence Position'
							#print sentPosition
							
							df.loc[inde,'Length'] = length
							df.loc[inde,'Stop Words'] = stopCount
							df.loc[inde,'Non Stop Words'] = nonStopCount
							df.loc[inde,'Pronouns'] = pronounCount
							df.loc[inde,'Position'] = float(1)/float(sentPosition)
							df.loc[inde,'Causal Signals'] = causalCount
							df.loc[inde,'Temporal Signals'] = temporalCount
							df.loc[inde,'Temporal Expressions'] = len(list(re.finditer(r'<TIMEX2>',taggedSen)))
							df.loc[inde,'Logical Signals'] = logicalCount
							df.loc[inde,'Sum TFIDF'] = tfidfSum
							df.loc[inde,'Avg TFIDF'] = avg
							df.loc[inde,'pos * TFIDF'] = tfidfpos
							df.loc[inde,'TF Top10'] = float(topTenCount)/float(length)
							df.loc[inde,'TF Top30'] = float(topThirtyCount)/float(length)
							df.loc[inde,'TF Top50'] = float(topFiftyCount)/float(length)
							df.loc[inde,'TF Top100'] = float(topHundredCount)/float(length)
							df.loc[inde,'Sum LogOdd'] = sumLogOdd	
							df.loc[inde,'Top LogOdd'] = maxLogOdd
							df.loc[inde,'Avg LogOdd'] = avgLogOdd
							df.loc[inde,'Cross Entropy'] = crossEntropy
							df.loc[inde,'Has Temporal Expr'] = hasTempExpr				
							index = index+1
						i = i+1
					print 'Data Frame:'
					print "Topic",f
					print "Date",d
					print df
					#print 'Labels Data Frame'
					#print sentLabels
					sentFrames.append(df)
					sentLabelFrames.append(sentLabels)

sentFeaturesdf = pd.concat(sentFrames)
sentLabelsdf = pd.concat(sentLabelFrames)
#print 'Final Features DF'
#print sentFeaturesdf
#print 'Final Labels df'
#print sentLabelsdf


