''' 
Utility class to merge all anotated corpus into, either :
    - A pandas Dataframe
    - A csv file
    - An xml file + csv containing annotation
'''
import os
import re
from lxml import etree

class Dataset_factory:
	
	def __init__(self):
		
		self.dirpath = os.path.dirname(os.path.realpath(__file__))
		self.data, self.target = self.initialize_data_target()

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
						with open()
							for doc in tree.xpath("/EXPORT/doc"):
								contenttag = doc.find("UD_TXT_SOURCE_FR")
								content = contenttag.findtext("value")
								target.append(content)
		return (data, target)
    		
	def createArff(self, filename):
		
		with open(self.dirpath+filename, 'a', encoding='utf-8') as arff :	
			arff.write("@relation allocine\n\n@attribute id\t\tNUMERIC\n@attribute "+
				"class\t{autre, critique, cine, critique_autre}\n@attribute"+
				" content\tSTRING\n\n@data\n")
		    for root, subdirs, files in os.walk("corpus/"):
		    	for f in files :
		    		with open(root+"/"+f, 'r', encoding='utf-8') as fi :
		    			for id, line in enumerate(fi.readlines()):
			                category = re.sub("corpus/", "", root)
			                category = re.sub("\\\mono", "", category)
			                category = re.sub("\\\multi", "", category)
			                line = str(id) + ', ' + category + ', ' + '"' + content + '"\n'
			                arff.write(line)
			                
	def createSKDataset(self):
	
		dataset = Sklearndataset(self.data, self.target)
		return dataset
		
class Sklearndataset:
	
	def __init__(self, data_, target_):
		self.data = data_
		self.target = target_
		                

if __name__ == '__main__':
	
	pass
                