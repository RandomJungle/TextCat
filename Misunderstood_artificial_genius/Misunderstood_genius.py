import os
import nltk
import re
import random
from math import log
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.text import Text 
from nltk import ConditionalFreqDist, FreqDist
from nltk.collocations import *

class Misunderstood_genius:
	
	root = "C:/Users/juliette.bourquin/Desktop/writers/"
	
	def __init__(self, master) :
		self.master = master
		self.reader = PlaintextCorpusReader(self.root+master, r'.*', encoding='utf-8')
		self.text = self.reader.words()
		
	'''
	write a text based on most probable word to appear after each word. Prone to looping
	'''
	def generate_model(self, word, num=50):
		bigrams = nltk.bigrams(self.text)
		cfdist = nltk.ConditionalFreqDist(bigrams)
		print(cfdist[word].pformat())
		for i in range(num):
			print(word, end=' ')
			word = cfdist[word].max()
		
	'''
	write a text based on a random choice of word that appear in collocation in master's work
	'''
	def text_generator(self, word, num=10):
		verse = ""
		bigrams = nltk.bigrams(self.text)
		cfdist = nltk.ConditionalFreqDist(bigrams)
		for i in range(num):
			verse += word + ' '
			word_collocates = []
			for w in cfdist[word] :
				word_collocates.append(w)
			word = random.choice(word_collocates)
		return verse
			
	'''
	write a poem with each verse starting with most commons words in master's work
	'''	
	def compose_standard_poem(self, length):
		poem =''
		all_word_dist = nltk.FreqDist(w.lower() for w in self.text)
		mostcommon= all_word_dist.most_common(length)
		for word in [x[0] for x in mostcommon if re.search('[a-zA-Z]', x[0]) is not None]:
			verse = self.text_generator(word, title=title)
			poem += verse + '\n'
		return poem
			
	'''
	write a poeam with each verse starting with random words from master's work
	'''
	def compose_random_poem(self, length):
		poem = ''
		for word in random.sample([x for x in self.text if re.search('[a-zA-Z]', x) is not None], length):
			verse = self.text_generator(word, title=title)
			poem += verse + '\n'
		return poem
	
	'''
	write a text that jumps to line after every n number of words, but is composed of one block only
	'''	
	def compose_structured_poem(self, length):
		final_work = ""
		first_word = random.choice([w.lower() for w in self.text if re.search('[a-zA-Z]', w) is not None])
		paragraph = self.text_generator(word=first_word, num=length)
		paragraphlist = paragraph.split(' ')
		for i in range(1,len(paragraphlist)) :
			final_work += paragraphlist[i] + ' '
			if i % 10 == 0 :
				final_work += '\n'
		return final_work
	
	'''
	find the best title to capture the essence of his work, through random search into words
	'''
	def find_title(self):
		first_word = random.choice([w.lower() for w in self.text if re.search('[a-zA-Z]', w) is not None])
		length = random.choice([1,2,3,4,5,6,7,8,9])
		title = self.text_generator(word=first_word, num=length)
		return title
			
	'''
	write a piece of text to a file, send it to everyone in town and wait for the letters of rejection
	'''
	def draft_manuscript(self, title, func, length):
		masterpiece = func(length)
		with open(self.root+title+'.txt', 'w', encoding='utf-8') as manuscript :
			manuscript.write(masterpiece)
			signature = re.sub("(^[a-zA-Z])(/)(.)*?", "\1", self.master.capitalize())
			manuscript.write('\n\n\t\t\t\t' + signature)
				
if __name__ == "__main__" : 
	
	spleener = Misunderstood_genius('collectif')
	work_title = spleener.find_title()
	spleener.draft_manuscript(title=work_title, func=spleener.compose_structured_poem, length=100)