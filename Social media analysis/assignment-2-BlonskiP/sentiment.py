from polyglot.downloader import downloader
from polyglot.text import Text
import pandas as pd
import demoji

def check_tweets(tweets_data, group_list):
    df = pd.DataFrame(tweets_data)
    columns = ['username', 'polarity']
    result = pd.DataFrame(columns=columns)
    print(tweets_data.count())
    for political_group in group_list:
        print(political_group)
        political_group_data = pd.DataFrame(columns=columns)
        filter = df['username'] == political_group
        bad_words = pd.DataFrame(columns=['badwords'])
        data = df.where(filter).dropna()
        print(data)
        for index, row in data.iterrows():
            text = row['tweet']
            noEmoji = demoji.replace(text, "")
            text = Text(noEmoji)
            polarity=0
            try:
                polarity = int(text.polarity)
            except ValueError as e:
                if not text.raw == '':
                    for w in text.words:
                        try:
                            polarity+=w.polarity
                        except ValueError as e:
                            #print(str(e)+' word: '+w)
                            try:
                                bad_words=bad_words.append(pd.DataFrame([w],columns=['badwords']))
                            except:
                                pass
            political_group_data = political_group_data.append(
                pd.DataFrame(data=[[political_group, polarity]], columns=columns))
        groupby = political_group_data.groupby(by=['username'],as_index=False).sum()
        result=result.append(groupby)
        print('end of '+political_group)
    print(result)
    return result, bad_words
    pass
