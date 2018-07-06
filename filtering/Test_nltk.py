import os
import nltk
import re
from math import log
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk.text import Text 
from nltk import ConditionalFreqDist, FreqDist
from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()
stopwordsdir = "C:/Projects/Allocine/stopwords/used"
stopwords = []
root = "C:/Projects/Allocine/corpus2/"
cats = ['cine', 'autre', 'critique', 'critique_a']
reader = CategorizedPlaintextCorpusReader(root, r'.*\.txt', cat_pattern=r'(\w+)/*', encoding='latin-1')

text_all = Text(reader.words())
text_cine = Text(reader.words(categories='cine'))
text_autre = Text(reader.words(categories='autre'))
text_critique = Text(reader.words(categories='critique'))
text_critique_a = Text(reader.words(categories='critique_a'))
texts_list = [text_cine, text_autre, text_critique, text_critique_a]

def remove_accents(text):
	text = re.sub("[àâäÄÂÀ]", "a", text)
	text = re.sub("[éèêëÈÊËÉ]", "e", text)
	text = re.sub("[ïîìÏÎÌ]", "i", text)
	text = re.sub("[öôòÖÔÒ]", "o", text)
	text = re.sub("[ùûüÜÛÙ]", "u", text)
	text = re.sub("[«»!/:;,\?•€%—\"\\^@\*\d\-\+\]\<>)\(\[]", " ", text)
	text = re.sub("[œŒ]", "oe", text)
	text = re.sub("[Ææ]", "ae", text)
	text = re.sub("[Çç]", "c", text)
	text = re.sub("( ){2,}", " ", text)
	return text

for file in os.listdir(stopwordsdir): 
	with open(stopwordsdir+'/'+file, 'r', encoding='utf-8') as stopf :
		for line in stopf.readlines() :     
			line = remove_accents(line)
			stopwords.append(re.sub('\n', '', line))
			
ponctuation = ['...','?','.',';','/',':','!','*','%','&','#','€','"',"'",'(','~','[','{','-','|','_','\\',')',']','}']
stopwords += ponctuation

def show_frequency():
	# ça c'est pour voir un peu les mots les plus fréquents dans les textes
	# et on peut s'amuser avec les n-grams aussi, pour quoi pas ?
	# permet notamment de remarquer des similarités de vocabulaire
	cfd = nltk.ConditionalFreqDist(
			(genre, word)
			for genre in reader.categories()
			for word in reader.words(categories=genre) if word.lower() not in stopwords)
	
	print("Cinéma : " + str(cfd['cine'].most_common(20)))
	print("Autres : " + str(cfd['autre'].most_common(20)))
	print("Critiques de films : " + str(cfd['critique'].most_common(20)))
	print("Critiques autres : " + str(cfd['critique_a'].most_common(20)))

# ça c'est un truc rigolo pour générer du texte automatique à partir de la fréquence conditionelle de distribution
def generate_model(cfdist, word, num=50):
	for i in range(num):
		print(word, end=' ')
		word = cfdist[word].max()

def model_from_bigrams():
	text = reader.words(categories='critique_a')
	bigrams = nltk.bigrams(text)
	cfd = nltk.ConditionalFreqDist(bigrams)
	generate_model(cfd, 'lui')

def show_best_tf_idf(reader):
	vocabulary = set(stemmer.stem(word.lower()) for word in reader.words())
	vocab = {}
	for category in reader.categories() :
		vocab_cat = {}
		for fileid in reader.fileids(categories=category) :
			for word in vocabulary :
				count = 0
				for fileid in reader.fileids():
					if word in [stemmer.stem(word.lower()) for word in reader.words(fileids=fileid)]:
						count += 1
				text = Text(stemmer.stem(word.lower()) for word in reader.words(fileids=fileid))
				tf = len([w for w in text if w == word]) # simple term frequency
				#tf = 1 if word in document, 0 otherwise # binary weight
				idf = log(len(reader.fileids())/count)
				# here we have the tf idf for each word of vocabulary in taht specific doc
				tfidf = tf * idf
				if word not in vocab_cat :
					vocab_cat[word] = tfidf
				else :
					vocab_cat[word] += tfidf
		# do something to add the category dict to the global dict
		print(vocab_cat)

def show_mostfrequent():
	for cat in cats :
		text = Text(stemmer.stem(word.lower()) for word in reader.words(categories=cat) if word.lower() not in stopwords)
		fdist = FreqDist(text)
		print(cat)
		print(text.collocations())
		print(sorted(w for w in set(text) if len(w) > 7 and fdist[w] > 7))
		print(type(fdist))
		for word in sorted(fdist) :
			if fdist[word] > 100 :
				print((str(word) + "->" + str(fdist[word]) + "; ").encode('utf-8'))
		print('\n--------------------------------\n')
		
show_best_tf_idf(reader)