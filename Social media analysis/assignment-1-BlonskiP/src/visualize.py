import pandas as pd
import plotly.express as px

def vizualize(covid_summary_df):
    fig = px.choropleth(covid_summary_df, locations="location",
                        color="sum",  # lifeExp is a column of gapminder
                        hover_name="text",  # column to add to hover information
                        color_continuous_scale=px.colors.sequential.Reds,
                        locationmode='country names',
                        animation_frame='date')
    fig.show()
    fig.write_html("interactive map.html")


