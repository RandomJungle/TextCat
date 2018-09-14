from __future__ import print_function

from pprint import pprint
from time import time
import logging

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import FrenchStemmer

# Uncomment the following to do the analysis on all the categories
root = "C:/Projects/Allocine/"
corpus_path = root+"corpus/"
data = datasets.load_files(corpus_path, encoding='utf-8')

###################### Best pipeline for SVC ######################

# SCORE : 0.851
'''

stemmer = FrenchStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

pipeline = Pipeline([
	('vect', CountVectorizer(strip_accents='ascii')),
	('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
	('clf', SVC(kernel='linear', decision_function_shape='ovo')),
	])
parameters = {
'vect__analyzer' : ('word', stemmed_words),
'vect__ngram_range' : ((1,1), (1,2), (2,3)),
'clf__kernel' : ('linear', 'rbf', 'sigmoid')
} 
'''

stemmer = FrenchStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

pipeline = Pipeline([
    ('vect', CountVectorizer(strip_accents='ascii')),
    ('tfidf', TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)),
    ('clf', MultinomialNB()),
    ])
parameters = {
'vect__analyzer' : ('word', stemmed_words),
'vect__ngram_range' : ((1,1), (1,3)),
'clf__fit_prior' : (True, False)
} 


##################### Best pipeline for NuSVC #####################

# SCORE : 0.800
'''
pipeline = Pipeline([
	('vect', CountVectorizer(analyzer='word', strip_accents='ascii', ngram_range=(1,1))),
	('tfidf', TfidfTransformer(norm=None)),
	('clf', NuSVC(kernel='rbf', decision_function_shape='ovr', probability=True)),
	])
parameters = {
'clf__kernel' : ('rbf', 'sigmoid', 'linear', 'poly'),
'clf__probability' : (True, False),
'clf__shrinking' : (True, False),
'clf__tol' : (1e-3, 1e-4),
'clf__decision_function_shape' : ('ovr', 'ovo'),
} # best gamma is 'auto' after evaluation
'''
################# Best pipeline for SGDClassifier #################

# SCORE : 0.873
'''
pipeline = Pipeline([
	('vect', CountVectorizer(strip_accents='ascii', ngram_range=(1,1))),
	('tfidf', TfidfTransformer(norm=None)),
	('clf', SGDClassifier(fit_intercept=True, loss='log', penalty='none', warm_start=False)),
	])
parameters = {
'clf__loss' : ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
'clf__penalty' : ('none', 'l2', 'l1', 'elasticnet'),
'clf__fit_intercept' : (True, False),
'clf__shuffle' : (True, False),
'clf__warm_start' : (True, False),
}
'''

################# Best pipeline for MLPClassifier #################

# SCORE : 0.870
'''
pipeline = Pipeline([
	('vect', CountVectorizer(strip_accents='ascii', ngram_range=(1,1))),
	('tfidf', TfidfTransformer(norm=None)),
	('clf', MLPClassifier(learning_rate='constant', activation='logistic', solver='lbfgs', shuffle=True, warm_start=False, early_stopping=False)),
	])
parameters = {
#'clf__activation' : ('identity', 'logistic', 'tanh', 'relu'),
#'clf__solver' : ('lbfgs', 'sgd', 'adam'),
#'clf__learning_rate' : ('constant', 'invscaling', 'adaptive'),
#'clf__shuffle' : (True, False),
#'clf__warm_start' : (True, False),
'clf__early_stopping' : (True, False),
}
'''

################### Best pipeline for LinearSVC ###################

# SCORE : 0.853
'''
pipeline = Pipeline([
	('vect', CountVectorizer(strip_accents='ascii', ngram_range=(1,1))),
	('tfidf', TfidfTransformer(norm=None)),
	('clf', LinearSVC(penalty='l2', fit_intercept=False, loss='squared_hinge', multi_class='ovr')),
	])
parameters = {
#'clf__penalty' : ('l1', 'l2'),
'clf__loss' : ('hinge', 'squared_hinge'),
'clf__multi_class' : ('ovr', 'crammer_singer'),
'clf__fit_intercept' : (True, False),
}
'''

############ Best pipeline for RandomForestClassifier #############

# SCORE : 0.760-0.780 (lot of random inside the random forest proverbialy)
'''
pipeline = Pipeline([
	('vect', CountVectorizer(strip_accents='ascii', ngram_range=(1,1))),
	('tfidf', TfidfTransformer(norm=None)),
	('clf', RandomForestClassifier(bootstrap=True, criterion='gini', oob_score=False, warm_start=True)),
	])
parameters = {
'clf__random_state' : (2,4,6,8,10,None),
'clf__criterion' : ('gini', 'entropy'),
#'clf__bootstrap' : (True, False),
'clf__oob_score' : (True, False),
'clf__warm_start' : (True, False),
}
'''

if __name__ == "__main__":
	# multiprocessing requires the fork to happen in a __main__ protected block
	# find the best parameters for both the feature extraction and the classifier
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
	print("Performing grid search...")
	print("pipeline:", [name for name, _ in pipeline.steps])
	print("parameters:")
	pprint(parameters)
	t0 = time()
	grid_search.fit(data.data, data.target)
	print("done in %0.3fs" % (time() - t0))
	print()
	
	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))