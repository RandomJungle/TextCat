import csv
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from lxml import etree
from sklearn.metrics import confusion_matrix

def draw_cm():
    
    global_result = {}
    true = []
    pred = []
    for i in range(1,5):
        with open('result'+str(i)+'.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader :
                key = int(str(i)+row[0])
                global_result[key] = (row[1], row[2])
                true.append(row[1])
                pred.append(row[2])
            
    classes = sorted(['autre','cine','critique','critique_a'])
    cw = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cw, index=classes, columns=classes)
    fig = plt.figure(figsize=(10,8))
    ax=plt.subplot(222)
    heatmap = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=200.0, fmt="d", cmap=sn.light_palette((210, 90, 60), input="husl"))
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('annotation')
    plt.xlabel('prédiction')
    fig.savefig('figure.png')
    
def model_corpus():
    
    for i in range(1,5):
        with open('result'+str(i)+'.csv', 'r', encoding='utf-8') as csvfile, open('export'+str(i)+'.xml', 'r', encoding='utf-8') as exportfile :
            reader = csv.reader(csvfile, delimiter='\t')
            tree = etree.parse(exportfile)
            for row in reader :
                key = row[0]
                for doc in tree.xpath("/EXPORT/doc"):
                    id = doc.get("docNum")
                    if id == key :
                        contenttag = doc.find("UD_TXT_SOURCE_FR")
                        if contenttag is not None :
                            content = contenttag.findtext("value")
                            with open('corpus/'+row[1]+'/'+str(i)+key+'.txt', 'a+', encoding='utf-8') as rf:
                                rf.write(content)
                            
model_corpus()