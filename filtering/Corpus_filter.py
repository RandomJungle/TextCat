import re
import os

root = "C:/Projects/Allocine/"
corpus_path = root+"corpus/"
neo_corpus_path = root+"neocorpus/"

stopwords = []

def initListStopwords(path):
	with open(path, 'r', encoding='utf-8') as stopf :
		for line in stopf.readlines() :     
			line = convertToASCII(line)
	        stopwords.append(re.sub('\n', '', line))

with open(root+'stopwords/persons_stopwords.txt', 'r', encoding='utf-8') as stopf :
    
	for line in stopf.readlines() :     
		
		stopwords.append(re.sub('\n', '', line))
        
for dir in os.listdir(corpus_path) :
	
	for file in os.listdir(corpus_path+dir) :
		
		content = ""
		
		with open(corpus_path+dir+"/"+file, 'r', encoding='utf-8') as tf :
			
			content = tf.readline().lower()
			
			for item in stopwords :
				
				content = content.replace(item, "")
				
			content = re.sub("( ){2,}", " ", content)
	
		with open(neo_corpus_path+dir+"/"+file, 'w', encoding='utf-8') as rf :
			
			rf.write(content)