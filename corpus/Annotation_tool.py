'''
Annotation tool that extract source and reception date plus content from all entries of an xml file. 
The source and date, as well as the ID, are used to name the file, which content is the text inside the "UD_TXT_SOURCE_FR" tag.
for each Id the user is prompted for input about where to place the file.
'''

from lxml import etree
import re

data = "C:/Projects/Allocine/exports/export_1_18-06-19.xml"
root = "C:/Projects/Allocine/corpus/"

tree = etree.parse(data)
for doc in tree.xpath("/EXPORT/doc"):
    #text = doc.findtext("/SO_KEY_lbl/value")
    sourcetag = doc.find("SO_KEY_lbl")
    id = doc.get("docNum")
    source = sourcetag.findtext("value")
    source = re.sub("\s", "_", source)
    date = doc.findtext("UM_DATE_REC")
    contenttag = doc.find("UD_TXT_SOURCE_FR")
    content = contenttag.findtext("value")
    
    if int(id) > 0:
        filename = source + "_" + date + "_" + id + ".txt"
        print(content)
        category = input("category of text [k1;k2;a1;a2;ka1;ka2;c1;c2] : ")
        if category == "k1" :
            dossier = "critique/mono/"
        elif category == "k2" :
            dossier = "critique/multi/"
        elif category == "c1" :
            dossier = "cine/mono/"
        elif category == "c2" :
            dossier = "cine/multi/"
        elif category == "a1" :
            dossier = "autre/mono/"
        elif category == "a2" :
            dossier = "autre/multi/"
        elif category == "ka1" :
            dossier = "critique_a/mono/"
        elif category == "ka2" :
            dossier = "critique_a/multi/"
        else :
            dossier = ""
        
        if category != "p" :
            with open(root+dossier+filename, 'w', encoding='utf-8') as rf :
                rf.write(content)
                print(filename)