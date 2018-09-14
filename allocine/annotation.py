import csv
import os
from tkinter import *
from lxml import etree

from sklearn.externals import joblib
from sklearn import preprocessing

class Evaluation_tool :
		
	def __init__ (self, file, modelpath):
		
		# File utilities
		self.dirpath = os.path.dirname(os.path.realpath(__file__))
		self.modelpath = self.dirpath + '\\model.joblib'
		self.exportpath = self.dirpath + '\\export2.xml'
		self.result_file = self.dirpath + '\\result2.csv'
		self.index = self.initialize_index()
		
		# Prediction elements
		self.pipeline = joblib.load(self.modelpath)
		self.machine_cat = ""
		self.texts = self.parse_xml(self.exportpath)
		self.report = {}
		
		# Tkinter elements
		self.root = Tk()
		self.top = Frame(self.root)
		self.bottom = Frame(self.root)
		self.middle = Frame(self.root)
		self.textframe = Text(self.root)
		self.scrollbar = Scrollbar(self.root)
		self.scrollbar.config(command=self.textframe.yview)
		self.textframe.config(yscrollcommand=self.scrollbar.set)
		self.categoryframe = Entry(self.root, font = "Arial 12", justify='center', width=50, bg='lightcyan1')

		
	def initialize_index(self):
		
		max = 0
		try :
			with open(self.result_file, 'r+') as csvfile :
				reader = csv.reader(csvfile, delimiter='\t')
				for row in reader:
					print(row)
					temp_index = int(row[0])
					if temp_index > max :
						max = temp_index
				if max > 0 :
					max += 1
		except FileNotFoundError : 
			pass
		return max
	
	def parse_xml(self, file):
		
		text_dict={}
		tree = etree.parse(file)
		content_list = []
		id_list = []
		for doc in tree.xpath("/EXPORT/doc"):
			ident = int(doc.get("docNum"))
			id_list.append(int(ident))
			contenttag = doc.find("UD_TXT_SOURCE_FR")
			content = contenttag.findtext("value")
			content_list.append(content)
		y_pred = self.pipeline.predict(content_list)
		predictions = list(zip(y_pred, content_list))
		for i in range(0,len(id_list)) :
			text_dict[i] = predictions[i]
		return text_dict
	
	def decode_machine_category(self, index):
		
		if index == 0 :
			category = "autre"
		elif index == 1 :
			category = "cine"
		elif index == 2 :
			category = "critique"
		elif index == 3 :
			category = "critique_a"
		return category
	
	def map_index_from_category(self, category):
	
		if category == "autre":
			index = 0
		elif category == "cine":
			index = 1
		elif category == "critique":
			index = 2
		elif category == "critique_a":
			index = 3
		return index
	
	def run(self):
		
		try :
			self.display_window()
		except KeyError :
			pass
		self.display_results()
	
	def display_results(self):
		
		self.top.pack(side=TOP)
		self.textframe.pack(in_=self.top, side=LEFT, fill=BOTH, expand=True)
		self.textframe.delete('1.0', END)
		result = self.calculate_metrics()
		self.textframe.insert(INSERT, result)
		self.root.mainloop()
		
	def calculate_metrics(self):
		
		# confusion matrix
		confusion_matrix = [[0,0,0,0,'\treconnu comme autre'],
							[0,0,0,0,'\treconnu comme ciné'],
							[0,0,0,0,'\treconnu comme critique de film'],
							[0,0,0,0,'\treconnu comme critique autre']]
		# metrics to calculate error rate
		error_num = 0
		row_num = 0
		# examining result file			
		with open(self.result_file) as csvfile:
			reader = csv.reader(csvfile, delimiter='\t')
			for row in reader:
				row_num += 1
				human_pred = row[1]
				machine_pred = row[2]
				if human_pred != machine_pred :
					error_num += 1
				human_index = self.map_index_from_category(human_pred)
				machine_index = self.map_index_from_category(machine_pred)
				confusion_matrix[machine_index][human_index] += 1
		readable_matrix = "autre\tciné\tcritique\tcritique_a <-- classification humaine\n"
		error_rate = error_num * 100 / row_num
		for row in confusion_matrix :
			for item in row :
				readable_matrix += str(item) + "\t"
			readable_matrix += '\n'
		result = "Matrice de confusion : \n" + readable_matrix + '\n\nTaux d\'erreur : ' + str(error_rate) + '%' 
		return result

	def display_window(self):
		
		self.top.pack(side=TOP, fill=BOTH, expand=YES)
		self.bottom.pack(side=BOTTOM, fill=BOTH)
		self.middle.pack()
		self.scrollbar.pack(in_=self.top, side=RIGHT, fill=Y)
		self.textframe.pack(in_=self.top, side=LEFT, fill=BOTH, expand=True)
		self.categoryframe.pack()
		text = self.texts[self.index][1]
		self.machine_cat = self.decode_machine_category(self.texts[self.index][0])
		a = Button(self.root, text="Article\nthème autre", height = 4, width = 20, font="Arial 12", command=lambda : self.classify("autre", text))
		b = Button(self.root, text="Article\nthème cinéma", height = 4, width = 20, font="Arial 12", command=lambda : self.classify("cine", text))
		c = Button(self.root, text="Critique\nde cinéma", height = 4, width = 20, font="Arial 12", command=lambda : self.classify("critique", text))
		d = Button(self.root, text="Critique\nd'autre chose", height = 4, width = 20, font="Arial 12", command=lambda : self.classify("critique_a", text))	
		quit = Button(self.root, text="Quitter", height = 4, width = 20, font="Arial 12", command=self.root.destroy)
		a.pack(in_=self.bottom, side=LEFT)
		b.pack(in_=self.bottom, side=LEFT)
		c.pack(in_=self.bottom, side=LEFT)
		d.pack(in_=self.bottom, side=LEFT)
		quit.pack(in_=self.bottom, side = RIGHT)
		# write text to frame
		self.textframe.insert(INSERT, text)
		# write text and category predicted to interface
		self.categoryframe.insert(INSERT, 'Ce texte a été classé comme : ' + self.machine_cat)
		while self.index < len(self.texts):
			self.root.update()
		# once annotation is complete, we destroy all the widgets, to prepare result screen
		self.categoryframe.destroy()
		self.scrollbar.destroy()
		a.destroy() 
		b.destroy()
		c.destroy()
		d.destroy()
		
	def classify(self, human_cat, text) :
		
		# clear text frame
		self.textframe.delete('1.0', END)
		self.categoryframe.delete(0, 'end')
		self.report[self.index] = [human_cat, self.machine_cat]
		with open(self.result_file, 'a+') as rf : 
			rf.write(str(self.index)+'\t'+human_cat+'\t'+self.machine_cat+'\n')
		self.index += 1
		self.machine_cat = self.decode_machine_category(self.texts[self.index][0])
		self.textframe.insert(INSERT, self.texts[self.index][1])
		self.categoryframe.insert(INSERT, 'Ce texte a été classé comme : ' + self.machine_cat)
		
'''
	extract the text content from an xml file
'''
def write_xml_segment(inputfile, outputfile, start, end):
	tree = ET.parse('export3.xml')
	root = tree.getroot()
	with open(outputfile, 'w', encoding='utf-8') as testfile :
		testfile.write('<?xml version="1.0" encoding="utf-8"?>\n<EXPORT>\n')
		for child in root:
			docnum = int(list(child.attrib.values())[0])
			result = ET.tostring(child, encoding='utf-8')
			if start <= docnum <= end :
				testfile.write(result.decode("utf-8"))
		testfile.write('</EXPORT>')
'''
	Divide a large xml export into smaller subset
	parameters: size => size of the files being processed (in number of xml docs)
	step => size of the files output (in number of xml docs)
'''	
def write_global_xml(step, size):
	for i in range(1,size,step):
		filename = 'export_' + str(random.randint(100,900)) + '.xml'
		write_xml_segment(filename, i, i+step)

if __name__ == "__main__" :
	
	test_file = "C:/Projects/Allocine/exports/export_tests.xml"
	modelpath = "C:/Projects/Allocine/model.joblib"
	evaluation = Evaluation_tool(test_file, modelpath)
	evaluation.run()