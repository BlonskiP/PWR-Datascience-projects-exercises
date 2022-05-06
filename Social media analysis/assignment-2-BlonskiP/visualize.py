import plotly.express as px
import pandas as pd
import plotly
import datetime

SELECTOR_ALL = dict(
    buttons=list([
        dict(count=1, label="1d", step="day", stepmode="backward"),
        dict(count=7, label="1w", step="day", stepmode="backward"),
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(step="all")
    ])
)


def visualize_tweets_all(tweet_data,hours):
    visualize_data = pd.DataFrame(tweet_data)

    visualize_data['date'] = pd.to_datetime(visualize_data['date'].astype(str) +' '+ visualize_data['time'].astype(str))
    visualize_data['date'] = visualize_data['date'].dt.floor(str(hours)+'H')
    visualize_data = visualize_data.groupby(by=['date', 'username'], as_index=False).count().sort_values(by=['date'])
    visualize_data = visualize_data.rename(columns={"id": "tweet count"})
    # print(visualize_data)
    fig = px.line(visualize_data, x='date', y="tweet count", color='username')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=SELECTOR_ALL
    )
    fig.show()
    return visualize_data

def visualize_tweets_values(tweet_data,hours,column):
    visualize_data = pd.DataFrame(tweet_data)

    visualize_data['date'] = pd.to_datetime(visualize_data['date'].astype(str) +' '+ visualize_data['time'].astype(str))
    visualize_data['date'] = visualize_data['date'].dt.floor(str(hours)+'H')
    visualize_data = visualize_data.groupby(by=['date', 'username'], as_index=False).sum().sort_values(by=['date'])
    # print(visualize_data)
    fig = px.line(visualize_data, x='date', y=column, color='username')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=SELECTOR_ALL
    )
    fig.show()
    return visualize_data
def visualize_column_sum(tweet_data,column):
    visualize_data = pd.DataFrame(tweet_data)
    visualize_data = visualize_data.groupby(by=['username'], as_index=False).sum()
    fig = px.bar(visualize_data, x='username', y=column)
    fig.show()

def visualize_column_count(tweet_data):
    visualize_data = pd.DataFrame(tweet_data)
    visualize_data = visualize_data.groupby(by=['username'], as_index=False).count()
    visualize_data = visualize_data.rename(columns={"id": "tweet count"})
    fig = px.bar(visualize_data, x='username', y='tweet count')
    fig.show()

def visualize_polarity(tweet_data):
    visualize_data = pd.DataFrame(tweet_data)
    fig = px.bar(visualize_data, x='username', y='polarity')
    fig.show()