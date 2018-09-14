''' 
Utility class to merge all anotated corpus into, either :
    - A pandas Dataframe
    - A csv file
    - An xml file + csv containing annotation
'''
import os
import re
from lxml import etree
from io import StringIO

class Dataset_factory:
    
    def __init__(self):
    	
    	self.dirpath = os.path.dirname(os.path.realpath(__file__))
    	#self.data, self.target = self.initialize_data_target()
    
    def initialize_data_target(self):
    	
        data = np.array()
        target = np.array()
        for dir in os.listdir(self.dirpath):
            if dir in ['Denitza','Aurore','Andrea','Odile','Juliette'] :
                for file in self.dirpath+'\\'+dir:
                    if file.endswith('.csv'):
                        with open(self.dirpath+'\\'+dir+'\\'+file, 'r', encoding='utf-8') as predfile :
                            reader = csv.reader(delimiter='\t')
                            for row in reader :
                                target.append(row[1])
                    if file.endswith('.xml'):
                        with open(self.dirpath+'\\'+dir+'\\'+file, 'r', econding='utf-8') as contentfile:
                            for doc in tree.xpath("/EXPORT/doc"):
                                contenttag = doc.find("UD_TXT_SOURCE_FR")
                                if contenttag is not None :
                                    content = contenttag.findtext("value")
                                    target.append(content)
        return (data, target)
               
    def createSKDataset(self):
    
    	dataset = Sklearndataset(self.data, self.target)
    	return dataset
    
    def extractFilesFromXMLExport(self, xmlfile, dir):
        
        tree = etree.parse(xmlfile)
        for doc in tree.xpath("/EXPORT/doc"):
            id = doc.get("docNum")
            date = doc.findtext("UM_DATE_REC")
            contenttag = doc.find("UD_TXT_SOURCE_FR")
            if contenttag is not None :
                content = contenttag.findtext("value")
                with open(dir+'/'+id+'_'+date+'.txt', 'w', encoding='utf-8') as rf:
                    rf.write(content)

class Sklearndataset:
	
	def __init__(self, data_, target_):
		self.data = data_
		self.target = target_
		                

if __name__ == '__main__':
	
	datafactory = Dataset_factory()
	#directory = "C:/Users/juliette.bourquin/Desktop/corpus"
	xmlfile = 'export.xml'
	datafactory.extractFilesFromXMLExport(xmlfile, datafactory.dirpath)       