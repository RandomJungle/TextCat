import csv
import os
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from builtins import dir

'''
Binary classification of different examples, and metric tools for evaluation
'''

class Binary_evaluator :
    
    def __init__ (self):
        
        # File utilities
        self.rootpath = "C:/Users/juliette.bourquin/Desktop/"
        self.corpuspath = self.rootpath+'corpus/'
        self.categories = sorted(['recettes', 'agenda', 'sport', 'tierce', 'tv'])
        
        # content
        self.content_list = []
        for file in os.listdir(self.corpuspath):
            with open(self.corpuspath+file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.content_list.append(content)

        # Dictionary to store results ?
        self.result = {}
        
    def extract_test_set(self):
        pass
        
    def binarize_corpus(left_class):
    
        for directory in os.listdir(corpus_path):
            if os.path.isdir(corpus_path+directory) and directory != left_class and directory != 'autres':
                for file in os.listdir(corpus_path+directory):
                    try:
                        os.rename(corpus_path+directory+'/'+file, corpus_path+'autres/'+file)
                    except FileExistsError :
                        random_id = random.randint(0,1000)
                        new_name = re.sub('.txt', str(random_id)+'.txt', file)
                        os.rename(corpus_path+directory+'/'+file, corpus_path+'autres/'+new_name)
                os.rmdir(corpus_path+directory)

    def run_models(self, contentlist, modelpath):
        
        pipeline = joblib.load(modelpath)
        y_pred = pipeline.predict(contentlist)
        return y_pred
    
    def reorganize_corpus(self):
        for directory in os.listdir(self.corpuspath):
            if os.path.isdir(self.corpuspath+directory):
                for file in os.listdir(self.corpuspath+directory+'/'):
                    new_name = directory + str(random.randint(0,10000000000))
                    os.rename(self.corpuspath+directory+'/'+file, self.corpuspath+new_name+'.txt')
                
    def global_evaluation(self):
        
        global_list = [[re.sub('(\d)*?\.txt', '', x)] for x in os.listdir(self.corpuspath)]
        for categorie in self.categories:
            binary_categories = [categorie, 'autres']
            enc = LabelEncoder().fit(binary_categories) 
            modelpath = self.rootpath+'models/'+categorie+'.joblib'
            prediction = self.run_models(self.content_list, modelpath)
            decoded = enc.inverse_transform(prediction)
            for i in range(0,len(global_list)):
                global_list[i].append(decoded[i])
        with open('result_file.csv', 'w', encoding='utf-8') as result :
            for list in global_list :
                if list[0] == 'faits_divers':
                    list[0] = 'autres'
                elif list[0] == 'petites_annonces':
                    list[0] = 'autres'
                elif list[0] == 'pub':
                    list[0] = 'autres'
                elif list[0] == 'horoscope':
                    list[0] = 'autres'
                result.write('\t'.join(list)+'\n')
        return global_list
    
    def read_data(self, datafile):
        
        global_list = []
        with open(datafile, 'r', encoding='utf-8') as data :
            for line in data.readlines():
                line = re.sub('\n', '', line)
                local_list = line.split('\t')
                global_list.append(local_list)
        return global_list
    
    def evaluate_metrics(self, result_list):
        
        error_count = 0
        valid_count = 0
        for line in result_list :
            category = line[0]
            for i in range(1,6):
                if category == line[i]:
                    valid_count += 1
                elif line[i] != 'autre':
                    error_count += 1
        print('number of erroneous predictions = ' + str(error_count))
        print('number of valid predictions = ' + str(valid_count))
        
    def confusion_matrix(self, result_list):
        
        cm = np.zeros((6, 6), dtype=int)
        def index(catego_string):
            categories = ['agenda', 'autres', 'recettes', 'sport', 'tierce', 'tv']
            return categories.index(catego_string)
        for line in result_list :
            category = line[0]
            predictions = line[1:]
            if category in predictions :
                cm[index(category)][index(category)] += 1
            elif len(set(predictions)) == 1 :
                cm[index(category)][index('autres')] += 1
            else :
                for pred in predictions :
                    if pred != 'autres' and pred != category :
                        cm[index(category)][index(pred)] += 1
        print(cm)
        return cm
        
    def show_matrix(self, cm):
        
        classes = ['agenda', 'autres', 'recettes', 'sport', 'tierce', 'tv']
        dataframe = pd.DataFrame(cm, index=classes, columns=classes)
        fig = plt.figure(figsize=(10,8))
        ax=plt.subplot(222)
        heatmap = sns.heatmap(dataframe, ax=ax, annot=True, fmt="d", cmap=sns.light_palette((210, 90, 60), input="husl"))
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('annotation')
        plt.xlabel('prédiction')
        fig.savefig('figure.png')

if __name__ == '__main__':
    
    bieval = Binary_evaluator()
    bieval.global_evaluation()
    results = bieval.read_data('result_file.csv')
    bieval.show_matrix(bieval.confusion_matrix(results))