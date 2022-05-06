# Assignment 2
## Participation of Polish Parties on Twitter

### Scope: knowledge and skils
* understanding Twitter data
* Twitter data acquisition
* basic textual analysis
* time series visualization 
* Docker/venv
* Jupiter ipython notebooks (or Jupiter Lab)
* Python:  
	* data organization: pandas
	* visualization: matplotlib, seaborn, bokeh, plotly
	* NLP: polyglot 
	* scrapping: twint, scrapy, newspaper3k
* Teamwork
<br><br>

### Tasks

1. (2 point) **Data.** 

Using appropriate python module acquire Twitter dataset about parliament parties activity on Twitter. Please consider the following parties (grouped by their coalition) that have been discovered by teaching team: 
```python
[pisorgpl, Porozumienie__, SolidarnaPL], 
[platforma_org, Nowoczesna, Zieloni, inicjatywaPL], 
[KONFEDERACJA_, RuchNarodowy, Partia_KORWiN], 
[nowePSL, KUKIZ15],
[__Lewica, partiarazem]
```
This is not the full list of parties, other were not found on Twitter (but we encourage to find them by your own if it's possible).

*Expected outcome:* a good quality raw dataset, with appropriate scope (see requirements in following tasks) for all candidates

<br>

2. (5 points) **Analysis & Visualization**:

A. Show the number of tweets in time (x axis) for parties using any line plot; show the number of re-tweets in time (x axis) for them using any line plot

B. Show the popularity of tweets in time (number of likes, number of retweets) - likes assigned to tweet's publication date

C. Show the distribution of activity in the scope of a day (on an hourly basis) and week (on a daily basis) for tweeting and retweeting activities (both)

D. Compare results of parties composing coalition. Does all of them follow similar scheme?

*Expected outcome:* a set of good quality visualization, easy to interpret, with appropriate axis description 

<br>

3. (4 points) **Sentiment analysis.** 

Using appropriate python module show visualization of tweet sentiment (average) in time for all parties. This is a simple evaluation of sentiment, no need to consider entities (aspects). Provide your explanation and objective interpretation. Check also the sentiment of comments (responses) to posts if you have collected appropiate data.

*Expected outcome:* a good quality visualization of sentiment distribution for all parties (posts and separatelly comments)

With this sentiment analysis we'd like to answer following questions:
1. Does ruling parties posts sentiment differ from opposition parties? Which are most polarized?
2. Does parties posts sentiment differ from people's comments? Which party is mostly loved and which is mostly hated?
3. Does parites forming a coallition speak with the same voice?


<br>

4. (3 points from project proposal pool) **Project proposal template fill part I**

Fill header + first 4 parts of project proposal form (Background, Scientific/technological questions of your project, Objectives of the project, The final product of the project)


<br><br>

### Readings
[TWINT scraper](https://github.com/twintproject/twint)

[TweetScraper project](https://github.com/jonbakerfish/TweetScraper)

[Build own Twitter scraper on Scrapy](https://amitupreti.com.np/posts/hands-on-web-scraping-building-your-own-twitter-dataset-with-python-and-scrapy/)

[Seaborn visualization lib](https://seaborn.pydata.org/)

[Polyglot sentiment analyzer](https://polyglot.readthedocs.io/en/latest/Sentiment.html)
