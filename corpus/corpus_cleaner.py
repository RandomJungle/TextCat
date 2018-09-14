import os

directory = os.path.dirname(os.path.realpath(__file__))+ '/'
new_dir = directory+'test/'

def delete_files():
    for file in os.listdir(directory):
        thisFile = directory+file
        print('\n'+thisFile+'\n')
        deleteFile = False
        with open(thisFile, 'r', encoding='utf-8') as f :
            print(f.read())
            text = input("is this Tiercé ?? [N/n : no] ")
            if text == 'N' or text == 'n':
                deleteFile = True
        if deleteFile :
            os.remove(thisFile)

def arrange_files():
    for file in os.listdir(directory):
        thisFile = directory+file
        if not os.path.isdir(thisFile) and thisFile.endswith('.txt'):
            print('\n'+thisFile+'\n')
            with open(thisFile, 'r', encoding='utf-8') as f :
                print(f.read())
            text = input("\nwhat is this ?? \nAutres = 0\nRecette = 1\nTierce = 2\nTV = 3\nSport = 4\nAgenda = 8\n\n")
            if text == '0':
                os.rename(thisFile, new_dir + 'autres/'+ file)
            elif text == 'pub':
                os.rename(thisFile, new_dir + 'pub' + file)
            elif text == 'faits':
                os.rename(thisFile, new_dir + 'faits_divers' + file)
            elif text == '1':
                os.rename(thisFile, new_dir + 'recettes/' + file)
            elif text == '2':
                os.rename(thisFile, new_dir + 'tierce/' + file)
            elif text == '3':
                os.rename(thisFile, new_dir + 'tv/' + file)
            elif text == '4':
                os.rename(thisFile, new_dir + 'sport/' + file)
            elif text == 'horoscope':
                os.rename(thisFile, new_dir + 'horoscope' + file)
            elif text == 'annonce':
                os.rename(thisFile, new_dir + 'petites_annonces' + file)
            elif text == '7':
                os.rename(thisFile, new_dir + 'cine/' + file)
            elif text == '8':
                os.rename(thisFile, new_dir + 'agenda/' + file)
            else :
                pass

def count_corpus():
    
    dict = {}
    for file in os.listdir(directory):
        if os.path.isdir(directory+file):
            counter = 0
            for subfile in os.listdir(directory+file):
                counter += 1
            dict[file] = counter
    for key, value in dict.items():
        print(key + ' : ' + str(value))
            
count_corpus()
arrange_files()