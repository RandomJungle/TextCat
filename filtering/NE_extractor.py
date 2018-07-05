from lxml import etree
import re

def incrementList(data) :
	tree = etree.parse(data)

	list_ne = []

	for doc in tree.xpath("//NE_SOCIETE | //NE_MARQUE | //NE_MARQUE_S"):
		text = doc.findtext("value")
		list_ne.append(text.lower())
	for doc in tree.xpath("//NE_PERSON | //NE_JOURNALIST | //NE_POLITICIAN |"):
		text = doc.findtext("value")
		text = re.sub(r"^([^ ]+?)( ){1}([^ ]+?)$",r"\3\2\1",text)
		list_ne.append(text.lower())
		
	return list_ne

def incrementListOnlyPersons(data) :
	tree = etree.parse(data)

	list_ne = []
 
	for doc in tree.xpath("//NE_PERSON | //NE_JOURNALIST | //NE_POLITICIAN |"):
		text = doc.findtext("value")
		text = re.sub(r"^([^ ]+?)( ){1}([^ ]+?)$",r"\3\2\1",text)
		list_ne.append(text.lower())
		
	return list_ne
	
global_list = []
	
for i in range(1,5) :
	data = "C:/Projects/Allocine/exports/export" + str(i) + ".xml"
	global_list.append(incrementListOnlyPersons(data))

flatten_list = [item for sublist in global_list for item in sublist]
final_list = set(flatten_list)

with open('C:/Projects/Allocine/stopwords/persons_stopwords.txt', 'w', encoding='utf-8') as f :
	for el in final_list :
	
		f.write(el + "\n")