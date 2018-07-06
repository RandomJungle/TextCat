from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.externals import joblib

import numpy as np
import re
import os

root = "C:/Projects/Allocine/"
corpus_path = root+"corpus2/"
dataset = datasets.load_files(corpus_path, encoding='utf-8')
stopwordsdir = "C:/Projects/Allocine/stopwords/used"
stopwords = []

for file in os.listdir(stopwordsdir) : 
	with open(stopwordsdir+'/'+file, 'r', encoding='utf-8') as stopf :
		for line in stopf.readlines() :     
			#line = remove_accents(line)
			stopwords.append(re.sub('\n', '', line))

def cross_validate(choice) : 

	if choice == 1 :
		classifier = SGDClassifier(fit_intercept=True, loss='log', penalty='none', warm_start=False)
	elif choice == 2 :
		classifier = MLPClassifier(learning_rate='constant', activation='logistic', solver='lbfgs', shuffle=True, warm_start=False, early_stopping=False)
	elif choice == 3 :
		classifier = LinearSVC(penalty='l2', fit_intercept=False, loss='squared_hinge', multi_class='ovr')
	elif choice == 4 :
		classifier = SVC(kernel='linear', decision_function_shape='ovo')
		
	vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', 
		analyzer='word', ngram_range=(1,3), stop_words=stopwords,
		lowercase=True, smooth_idf=True, max_df=0.8, min_df=2)

	text_clf = Pipeline([('vect', vectorizer),
		('clf', classifier)])
	
	scores = cross_val_score(text_clf, dataset.data, dataset.target, cv=5)
	print("Classifier : \n" + str(classifier) + "\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def remove_accents(text) :
	
	text = re.sub("[àâäÄÂÀ]", "a", text)
	text = re.sub("[éèêëÈÊËÉ]", "e", text)
	text = re.sub("[ïîìÏÎÌ]", "i", text)
	text = re.sub("[öôòÖÔÒ]", "o", text)
	text = re.sub("[ùûüÜÛÙ]", "u", text)
	return text

def tokenize(content) :
	
    content = re.sub("[\"'\n\t,\.«»•!\?/\\\(\)_\-\^&\*:;%~]", " ", content)
    content = re.sub("(\s){2,}", " ", content)
    return content

def remove_stopwords(content) :

    content = content.split(" ")
    contentList = [w for w in content if w not in stopwords]
    content = ' '.join(contentList)
    #To remove numerical values
    content = re.sub("[0-9]", "", content)
    content = re.sub("(\s){2,}", " ", content)
    return content

def classic_prediction(choice) :
	
	if choice == 1 :
		classifier = SGDClassifier(fit_intercept=True, loss='log', penalty='none', warm_start=False)
	elif choice == 2 :
		classifier = MLPClassifier(learning_rate='constant', activation='logistic', solver='lbfgs', shuffle=True, warm_start=False, early_stopping=False)
	elif choice == 3 :
		classifier = LinearSVC(penalty='l2', fit_intercept=False, loss='squared_hinge', multi_class='ovr')
	elif choice == 4 :
		classifier = SVC(kernel='linear', decision_function_shape='ovo')
		
	vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', 
		analyzer='word', ngram_range=(1,3), stop_words=stopwords,
		lowercase=True, smooth_idf=True, max_df=0.8, min_df=2)

	text_clf = Pipeline([('vect', vectorizer),
		('clf', classifier)])

	X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=0)
	
	text_clf.fit(X_train, y_train) 
	predicted = text_clf.predict(X_test)
	print("Score pour le classifieur " + str(classifier) + " : \n" + str(np.mean(predicted == y_test)) + "\n")
	print(metrics.classification_report(y_test, predicted, target_names=dataset.target_names))
	print(metrics.confusion_matrix(y_test, predicted))

def predict_and_save_pipeline():

	pipeline = Pipeline([
		('vect', CountVectorizer(analyzer='word', strip_accents='ascii', ngram_range=(1,1))),
		('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
		('clf', SVC(kernel='linear', decision_function_shape='ovo')),
		])
		
	# train the model
	pipeline.fit(dataset.data, dataset.target)
	
	# Export the classifier to a file
	joblib.dump(pipeline, "C:/Projects/Allocine/model.joblib")

if __name__ == '__main__':
	
	#for i in [1,2,3,4] :
	#	cross_validate(i)
	predict_and_save_pipeline()