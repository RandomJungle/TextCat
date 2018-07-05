''' 
Utility class to merge all anotated corpus into, either :
    - A pandas Dataframe
    - A csv file
    - An xml file + csv containing annotation
'''

class Corpus_merger:

    def __init__(self):
    	
    	self.data = self.read_files()

    '''
    Dictionary meant to contain a unique ID, composed of the anotator_name + fileid, the text, the machine annotation and the human annotation
    '''
    def read_files(self):

        dict = {}
        return dict

	'''
	write method to encode data into file (might need to split that for when needed different type of file ?
	'''
    def write_data(self):

        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(self.data)

if __name__ == '__main__':
	
	merger = Corpus_merger()
	merger.write_data()