from bs4 import BeautifulSoup

def extract_content(file) :
	
	with open(file, "r", encoding='utf-8') as f:
		
		soup = BeautifulSoup(f, 'lxml')   # can be changed to 'lxml'
	
		for td in soup.find_all('td') :
		
			if not td.a is None :
				
				print(td.a.string)
		
file = "C:/Projects/Allocine/exports_wikipedia/french_towns.html"

extract_content(file)