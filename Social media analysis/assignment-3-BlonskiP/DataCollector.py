import twint
import os
import pandas as pd
from tqdm import tqdm
from NetworkCreator import Network_creator
TAG = ""
DIR_NAME = 'twits'
POLISH_LANGAUGE_CODE = 'pl'
COLS = ['username', 'followers']


def get_Twits_Authors(tag, twit_since_date, reset=True):
    PATH = os.path.join(DIR_NAME, tag)
    FILENAME = PATH + '.json'

    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    print(FILENAME)
    if not os.path.exists(FILENAME) or reset:
        print('Scraping twitts of tag: #' + tag)
        c = twint.Config()
        # Location, lang tag etc
        c.Limit = 200 #Nodes limit
        c.User_full = True
        c.Search = "#" + tag
        c.Lang = POLISH_LANGAUGE_CODE
        c.Since = twit_since_date
        # Output config
        # Pickle has only: ID,Date,hour,user,timezone,twit content
        # c.Store_pandas = True
        # c.Pandas_type = 'Pickle'
        # c.Output = 'twits/'"tag+".pyc"
        # Json has : id, conversation_id, created date,date/time/timezone,userid,username,name,twitcontnent,languagem netions and far more.
        # it will be better for this operation that pickle.

        c.Store_json = True
        c.Output = 'twits/' + tag + '.json'

        c.Hide_output = True
        twint.run.Search(c)

        print('Postmakers Scrapping DONE')
    else:
        print(FILENAME + " already exists.")


def get_Followers(username,usernames):
    try:
        c = twint.Config()
        #c.Limit = 10  # There is no time for full scaping (12h was not enouth)
        c.Hide_output = True
        c.Pandas = True
        c.Username = username
        twint.run.Followers(c)
        followers = twint.storage.panda.Follow_df
        #print('followers', followers)
        list_of_followers2 = []
        list_of_followers = followers['followers'][username]
        for follower in list_of_followers:
            if follower in usernames['username'].values:
                print(follower,'is connected to',username)
                list_of_followers2.append(follower)
        followers_of_user = pd.DataFrame([[username, list_of_followers2]], columns=COLS)
        return followers_of_user
    except:
        return None



def load_json(tags, columns):
    return_data = pd.DataFrame(columns=columns)
    for file in tags:
        PATH = os.path.join(DIR_NAME, file + '.json')
        # print(PATH)
        if os.path.isfile(PATH):
            print('loading :' + PATH)
            loaded_data = pd.read_json(PATH, lines=True)
            tmpData = loaded_data[columns]
            return_data = return_data.append(tmpData)
            #print(return_data.count())
    #print(return_data.count())
    PATH_PICKLE = os.path.join(DIR_NAME, file + '_users.csv')
    return_data=return_data.drop_duplicates(subset=['username'])
    return_data.to_csv(PATH_PICKLE)
    print('USERS_LEN '+str(return_data.size))
    return return_data


def get_all_Followers(usernames, tag):
    PATH_PICKLE = os.path.join(DIR_NAME, tag + '_follow_users.csv')
    # print(PATH)
    followers_df = pd.DataFrame(columns=COLS)
    if os.path.exists(PATH_PICKLE):
        followers_df = pd.read_csv(PATH_PICKLE)
    for index, row in tqdm(usernames.iterrows()):
        username=row["username"]
        print(username)
        if username not in followers_df['username'].values:
            followers=get_Followers(username,usernames)
            if followers is not None:
                followers_df=followers_df.append(followers)
                followers_df.to_csv(PATH_PICKLE)
            else:
                data = [username,[]]
                df = pd.DataFrame([data],columns=COLS)
                followers_df = followers_df.append(df)
                followers_df.to_csv(PATH_PICKLE)
    return followers_df


def collect_all(tag, twit_since_date, reset):
    # GET JSONS
    get_Twits_Authors(tag, twit_since_date, reset)
    # LOAD USERS WHO POSTED TWIT
    users = load_json([tag], ['username']).drop_duplicates()
    # Get followers of these users
    df = get_all_Followers(users, tag)

    return df

def get_full_user_data(username):
    c = twint.Config()
    c.Username = username

    c.User_full  = True
    c.Profile_full = True
    c.Hide_output = True
    c.Pandas = True
    twint.run.Search(c)
    userdata = twint.storage.panda.Follow_df
    print(userdata)
    return userdata

def create_users_group_dataset(node_groups):
    COLUMNS = ['username','Group']
    users_groups = pd.DataFrame(columns=COLUMNS)
    group_number = 0
    for group in node_groups:
        for name in group:
            user_data  = get_full_user_data(name)
            data = [name,group_number]
            #users_groups=users_groups.append(pd.DataFrame([data],columns=COLUMNS))
        group_number+=1



    pass