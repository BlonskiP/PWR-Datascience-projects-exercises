import networkx as nx
from ast import literal_eval
from NetworkPloter import visualize
import pandas as pd
import twint
import os
import time
import numpy as np
from tqdm import tqdm

USER_COLUMNS = ['username', 'Group']
user_data_columns = ['timezone', 'nretweets', 'nlikes', 'geo', 'hour', 'place', 'count', 'likesCount']


def preproces_user_data(user_data):
    try:

        user_timezone = user_data.timezone.mode().at[0]
        sums = user_data.groupby(by='username').sum()
        count = user_data.groupby(by='username').count()
        nretweets = sums['nretweets'].mode().at[0]
        nlikes = sums['nlikes'].mode().at[0]
        count = count['id'].mode().at[0]
        Count_to_nlikes = nlikes / count
        geo = user_data.geo.mode().at[0]
        hour = user_data.hour.mode().at[0]  # when most posts accour
        place = user_data.place.mode().at[0]  # from where
        user_arr = [user_timezone, nretweets, nlikes, geo, hour, place, count, Count_to_nlikes]
        df = pd.DataFrame([user_arr], columns=user_data_columns)
        return df
    except Exception as e:
        print(e)
        return None


def get_full_user_data(username):
    print('scraping', username)
    new_data = None
    try:
        c = twint.Config()
        # twint.token.Token(c).refresh() doesn't work
        c.Username = username
        c.Profile_full = True
        c.User_full = True
        c.Hide_output = True
        c.Pandas = True
        c.Since = '2020-01-01'
        POLISH_LANGAUGE_CODE = 'pl'
        c.Lang = POLISH_LANGAUGE_CODE
        twint.run.Search(c)
        userdata = twint.storage.panda.Tweets_df
        new_data = preproces_user_data(userdata)
    except KeyError as e:
        print('ERROR FOR', username, e)
        return None

    if new_data is not None:
        return new_data
    else:
        return None


def create_users_group_dataset(node_groups):
    COLUMNS = ['username', 'timezone', 'nretweets', 'nlikes', 'geo', 'hour', 'place', 'count', 'likesCount', 'group']
    users_groups = pd.DataFrame(columns=COLUMNS)
    group_number = 0
    ppl = 0
    PATH = 'Users_dataset.csv'
    if os.path.isfile(PATH):
        users_groups = pd.read_csv(PATH)
    print("aLl groups: ",len(node_groups))
    for group in tqdm(node_groups):
        for name in tqdm(group):
            if name not in users_groups['username'].values:
                data = pd.DataFrame([[name]], columns=['username'])
                user_data = get_full_user_data(name)
                if user_data is not None:
                    print(ppl)
                    ppl += 1
                    result = pd.concat([data, user_data], axis=1, sort=False)
                    group_number_df = pd.DataFrame([group_number], columns=['group'])
                    result = pd.concat([result, group_number_df], axis=1, sort=False)
                    users_groups = users_groups.append(result)
                    users_groups.to_csv(PATH, index=False)
            else:
                print(name,'group update to', group_number)
                users_groups['group'] = users_groups.apply(lambda x: group_number if x['username']==name else x['group'], axis=1)
        group_number += 1
    users_groups.to_csv(PATH, index=False)
    pass


class Network_creator:
    def create_network(self, users, tag):
        print(type(users['followers']))
        G = nx.Graph()
        for index, row in users.iterrows():
            user = row['username']
            followers = row['followers']
            if followers != '[]':
                followers = literal_eval(followers)
            else:
                followers = []
            for follower in followers:
                G.add_edge(user, follower)
        nx.write_edgelist(G, path=tag + ".edgelist", delimiter=":")
        return G

    # GirvanNewman Method for community detection.
    # Source: https://www.analyticsvidhya.com/blog/2020/04/community-detection-graphs-networks/
    def edge_to_remove_GirvanNewman(self, graph):
        G_dict = nx.edge_betweenness_centrality(graph)
        edge = ()
        # extract the edge with highest edge betweenness centrality score
        for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse=True):
            edge = key
            break
        return edge

    def girvan_newman(self, graph):
        print('girvan_newman RUN')
        # find number of connected components
        sg = nx.connected_components(graph)
        sg_count = nx.number_connected_components(graph)
        MAX_DEPTH=5
        while (graph.number_of_edges()>0 and sg_count<MAX_DEPTH):
            graph.remove_edge(self.edge_to_remove_GirvanNewman(graph)[0], self.edge_to_remove_GirvanNewman(graph)[1])
            sg = nx.connected_components(graph)
            sg_count = nx.number_connected_components(graph)
        print('girvan_newman DONE',sg_count)
        return sg

    def get_community_network(self, users, tag):
        followers_network = self.create_network(users, tag)
        community_network = self.girvan_newman(followers_network.copy())
        node_groups = []
        for i in community_network:
            node_groups.append(list(i))
        create_users_group_dataset(node_groups)
        visualize(followers_network, node_groups)
