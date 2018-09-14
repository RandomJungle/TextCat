import numpy as np
import pandas as pd
import seaborn as sn
import re
import os
import random
import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import FrenchStemmer

from imblearn.over_sampling import SMOTE, ADASYN

class Categorizer :
	
	def __init__(self, class_labels, corpus_path):
		
		self.stemmer = FrenchStemmer()
		self.analyzer = CountVectorizer().build_analyzer()
		
		self.class_labels = class_labels
		self.corpus_path = corpus_path
		self.dataset = datasets.load_files(self.corpus_path, encoding='utf-8', categories=self.class_labels)
		stopwordsdir = "C:/Projects/Allocine/stopwords/used"
		self.stopwords = []
		for file in os.listdir(stopwordsdir) : 
			with open(stopwordsdir+'/'+file, 'r', encoding='utf-8') as stopf :
				for line in stopf.readlines() :     
					#line = remove_accents(line)
					self.stopwords.append(re.sub('\n', '', line))

	def stemmed_words(self, doc):
	    return (self.stemmer.stem(w) for w in self.analyzer(doc))					

	def vectorize(self, X):

		vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', 
			analyzer=self.stemmed_words, ngram_range=(1,1), stop_words=self.stopwords,
			lowercase=True, smooth_idf=True, max_df=0.8, min_df=2)
		X_vectorized = vectorizer.fit_transform(X)
		return X_vectorized
			
	def resample_dataset(self, X, y):	
			
		style1 = 'borderline1'
		style2 = 'borderline2'
		style3 = 'svm'
		X_resampled, y_resampled = SMOTE(kind=style3).fit_sample(X, y)
		return X_resampled, y_resampled

	def cross_validate_pipeline(self, choice, X, y) : 
	
		if choice == 1 :
			classifier = SGDClassifier(fit_intercept=True, loss='log', penalty='none', warm_start=False)
		elif choice == 0 :
			classifier = MLPClassifier(learning_rate='constant', activation='logistic', solver='lbfgs', shuffle=True, warm_start=False, early_stopping=False)
		elif choice == 2 :
			classifier = LinearSVC(penalty='l2', fit_intercept=False, loss='squared_hinge', multi_class='ovr')
		elif choice == 3 :
			classifier = SVC(kernel='linear', decision_function_shape='ovo')
		elif choice == 5 :
			classifier = MultinomialNB()
		elif choice == 4 :
			classifier = MultinomialNB(alpha=1.0, fit_prior=True, analyzer='word')
			
		vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', 
			analyzer=self.stemmed_words, ngram_range=(1,3), stop_words=self.stopwords,
			lowercase=True, smooth_idf=True, max_df=0.8, min_df=2)
	
		text_clf = Pipeline([('vect', vectorizer),
			('clf', classifier)])
		
		scores = cross_val_score(text_clf, X, y, cv=10)
		print("Classifier : \n" + str(classifier) + "\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	
	def oversample_predict(self, choice, X, y):
		
		X_vectorized = self.vectorize(X)
		
		if choice == 4 :
			classifier = SGDClassifier(fit_intercept=True, loss='log', penalty='none', warm_start=False)
		elif choice == 2 :
			classifier = LinearSVC(penalty='l2', fit_intercept=False, loss='squared_hinge', multi_class='ovr')
		elif choice == 3 :
			classifier = SVC(kernel='linear', decision_function_shape='ovo')
		elif choice == 0 :
			classifier = MultinomialNB()
		elif choice == 1 :
			classifier = MultinomialNB(alpha=1.0, fit_prior=True)
		
		X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=0)
		X_train_resampled, y_train_resampled = self.resample_dataset(X_train, y_train)
		
		classifier.fit(X_train_resampled, y_train_resampled) 
		predicted = classifier.predict(X_test)
		print("Score pour le classifieur " + str(classifier) + " : \n" + str(np.mean(predicted == y_test)) + "\n")
		print(metrics.classification_report(y_test, predicted))
		print(metrics.confusion_matrix(y_test, predicted))
		clf_name = classifier.__class__.__name__
		return clf_name, metrics.confusion_matrix(y_test, predicted)

	def prediction_pipeline(self, choice, X_train, X_test, y_train, y_test) :
		
		if choice == 4 :
			classifier = SGDClassifier(fit_intercept=True, loss='log', penalty='none', warm_start=False)
		elif choice == 0 :
			classifier = MLPClassifier(learning_rate='constant', activation='logistic', solver='lbfgs', shuffle=True, warm_start=False, early_stopping=False)
		elif choice == 2 :
			classifier = LinearSVC(penalty='l2', fit_intercept=False, loss='squared_hinge', multi_class='ovr')
		elif choice == 3 :
			classifier = SVC(kernel='linear', decision_function_shape='ovo')
		elif choice == 5 :
			classifier = DecisionTreeClassifier() #MultinomialNB()
		elif choice == 1 :
			classifier = GaussianNB()#MultinomialNB(alpha=1.0, fit_prior=True)
		elif choice == 7 :
			classifier = SVC(kernel='linear', decision_function_shape='ovo', class_weight='balanced')
			
		vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', 
			analyzer=self.stemmed_words, ngram_range=(1,3), stop_words=self.stopwords,
			lowercase=True, smooth_idf=True, max_df=0.8, min_df=2)
	
		text_clf = Pipeline([('vect', vectorizer),
			('clf', classifier)])
	
		#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
		
		text_clf.fit(X_train, y_train) 
		predicted = text_clf.predict(X_test)
		print(text_clf.classes_)
		print("Score pour le classifieur " + str(classifier) + " : \n" + str(np.mean(predicted == y_test)) + "\n")
		print(metrics.classification_report(y_test, predicted))
		print(metrics.confusion_matrix(y_test, predicted))
		clf_name = classifier.__class__.__name__
		return clf_name, metrics.confusion_matrix(y_test, predicted)
		
	def one_class_prediction(self, choice, X, y):
		
		if choice == 1 :
			classifier = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
			
		vectorizer = TfidfVectorizer(encoding='utf-8', strip_accents='ascii', 
			analyzer=self.stemmed_words, ngram_range=(1,3), stop_words=self.stopwords,
			lowercase=True, smooth_idf=True, max_df=0.8, min_df=2)
	
		text_clf = Pipeline([('vect', vectorizer),
			('clf', classifier)])
	
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
		
		text_clf.fit(X_train, y_train)
		predicted = text_clf.predict(X_test)
		print("Score pour le classifieur " + str(classifier) + " : \n" + str(np.mean(predicted == y_test)) + "\n")
	
	def predict_and_save_pipeline(self, chosen_class, X, y):
	
		pipeline = Pipeline([
			('vect', CountVectorizer(analyzer=self.stemmed_words, strip_accents='ascii', ngram_range=(1,1))),
			('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
			('clf', SVC(kernel='linear', decision_function_shape='ovo')),
			])
		
		pipeline2 = Pipeline([
			('tfidf', TfidfVectorizer(strip_accents='ascii', analyzer=self.stemmed_words, ngram_range=(1,3), stop_words=stopwords, smooth_idf=True, max_df=0.8, min_df=2)),
			('clf', SGDClassifier(fit_intercept=True, loss='log', penalty='none', warm_start=False))
			])
			
		# train the model
		pipeline2.fit(X, y)
		
		# Export the classifier to a file
		joblib.dump(pipeline2, corpus_path+chosen_class+".joblib")
		
	def binarize_corpus(self, left_class):
		
		for directory in os.listdir(self.corpus_path):
			if os.path.isdir(self.corpus_path+directory) and directory != left_class and directory != 'autres':
				for file in os.listdir(self.corpus_path+directory):
					try:
						os.rename(self.corpus_path+directory+'/'+file, self.corpus_path+'autres/'+file)
					except FileExistsError :
						random_id = random.randint(0,1000)
						new_name = re.sub('.txt', str(random_id)+'.txt', file)
						os.rename(self.corpus_path+directory+'/'+file, self.corpus_path+'autres/'+new_name)
				os.rmdir(self.corpus_path+directory)
					
if __name__ == '__main__':
	
	nature_contenu_cats = sorted(['sport', 'agenda', 'autres', 'recettes', 'tv', 'tierce'])
								 #'pub', 'faits_divers', 'horoscope'])
	allocine_cats = sorted(['autre', 'cine', 'critique_a', 'critique'])
	
	#root = "../"
	#corpus_path = root+"nature_contenu/corpus-total/"
	#train_path = root+"nature_contenu/trainset/"
	#test_path = root+"nature_contenu/testset/"
	corpus_path = '../allocine/corpus/'
	categorizer = Categorizer(allocine_cats, corpus_path)

	fig = plt.figure(figsize=(10,8))
	fig.subplots_adjust(hspace=0.3, wspace=0.3)
	
	'''
	Xy_train = datasets.load_files(train_path, encoding='utf-8', categories=categorizer.class_labels)
	Xy_test = datasets.load_files(test_path, encoding='utf-8', categories=categorizer.class_labels)
	
	X_train = Xy_train.data
	X_test = Xy_test.data
	y_train = Xy_train.target
	y_test = Xy_test.target 
	'''
	
	for i in range(1,5) :
		
		#categorizer.cross_validate_pipeline(i, categorizer.dataset.data, categorizer.dataset.target)
		clf, cm = categorizer.oversample_predict(i, categorizer.dataset.data, categorizer.dataset.target)
		#clf, cm = categorizer.prediction_pipeline(i, X_train, X_test, y_train, y_test)
		df_cm = pd.DataFrame(cm, index=categorizer.class_labels, columns=categorizer.class_labels)
		ax = fig.add_subplot(2, 2, i)
		ax.text(0, 0, s=clf, ha='center', va='bottom')
		heatmap = sn.heatmap(df_cm, ax=ax, annot=True, fmt="d", vmin=0.0, vmax=100.0, cmap=sn.light_palette((210, 90, 60), input="husl"))
		heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
		heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
		
	fig.savefig('cm_figure.png')