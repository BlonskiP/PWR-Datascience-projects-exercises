import twint
import os
import pandas as pd

DIR_NAME = 'twits'


def getTwits(username, reset=True):
    PATH = os.path.join(DIR_NAME, username)
    FILENAME = PATH + '.json'

    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    print(FILENAME)
    if not os.path.exists(FILENAME) or reset:
        print('Scraping twitts of: ' + username)
        c = twint.Config()
        c.Username = username
        c.Store_json = True
        c.Since = '2018-01-01'
        c.Output = 'twits/' + username + '.json'
        c.Hide_output = True
        twint.run.Search(c)
        print('Scrapping DONE')
    else:
        print(FILENAME + " already exists.")


def Load_Jsons(file_names):
    columns = ['id','username','tweet','date','likes_count','retweets_count','time']
    return_data = pd.DataFrame(columns=columns)
    for file in file_names:
        PATH = os.path.join(DIR_NAME, file+'.json')
        if os.path.isfile(PATH):
            #print('loading :'+PATH)
            loaded_data = pd.read_json(PATH, lines=True)
            #print(loaded_data)
            tmpData = loaded_data[columns]
            tmpData['username']=file
            return_data=return_data.append(tmpData)
            print(return_data.count())
    print(return_data.count())
    return return_data


