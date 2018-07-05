import os
import re

i = 1
stopwords = []

def convertToASCII (text) :
    text = re.sub("[àâäÄÂÀ]", "a", text)
    text = re.sub("[éèêëÈÊËÉ]", "e", text)
    text = re.sub("[ïîìÏÎÌ]", "i", text)
    text = re.sub("[öôòÖÔÒ]", "o", text)
    text = re.sub("[ùûüÜÛÙ]", "u", text)
    # not sure this should go...
    text = re.sub("[ç]", "c", text)
    return text

def removeNumericalValues (text) :
    text = re.sub("[0-9]", "", text)
    return text

with open('stopwords.txt', 'r', encoding='utf-8') as stopf :
    
    for line in stopf.readlines() :     
        line = convertToASCII(line)
        stopwords.append(re.sub('\n', '', line))
 
with open('allocine.arff', 'a', encoding='utf-8') as arff :
    
    arff.write("@relation allocine\n\n@attribute id\t\tNUMERIC\n@attribute "+
               "class\t{autre, critique, cine, critique_autre}\n@attribute"+
               " content\tSTRING\n\n@data\n")

    for root, subdirs, files in os.walk("corpus/"):
    
        for f in files :
            
            with open(root+"/"+f, 'r', encoding='utf-8') as fi :
                
                content = fi.readlines()
                # delete any newline found in doc
                content = ' '.join(content)
                #to lower case
                content = content.lower()
                # remove all accents
                content = convertToASCII(content)
                # remove all punctuation
                content = re.sub("[\"'\n\t,\.«»•!\?/\\\(\)_\-\^&\*:;%~]", " ", content)
                content = re.sub("(\s){2,}", " ", content)
                # remove stopwords
                content = content.split(" ")
                contentList = [w for w in content if w not in stopwords]
                content = ' '.join(contentList)
                #○ remove numerical values
                content = removeNumericalValues(content)
                content = re.sub("(\s){2,}", " ", content)
                
                category = re.sub("corpus/", "", root)
                category = re.sub("\\\mono", "", category)
                category = re.sub("\\\multi", "", category)
                
                line = str(i) + ', ' + category + ', ' + '"' + content + '"\n'
                arff.write(line)
                i+=1
                