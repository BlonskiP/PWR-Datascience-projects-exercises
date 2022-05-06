# Assignment 3
## User communities in Twitter controversies

### Scope: knowledge and skils
* understanding Twitter data
* Twitter data acquisition
* Jupiter ipython notebooks (or Jupiter Lab)
* python:  
	* data organization: pandas
	* graphs: networkx, igraph, graph-tool
	* machine learning: scikit-learn, xgboost
	* NLP: gensim, spaCy, tweet2vec, polyglot 
	* scrapping: scrapy, twint
* teamwork
<br><br><br><br>
### Tasks

1. (2 point) 
**Data.** 
Using appropriate Python module acquire Twitter dataset (with all features that allows accomplishment of all three tasks) pertaining to a controversy of choice, identified through a hashtag (eg. #opos, #MuremzaRomanem, #Giertych #AferaSzumowskiego, #bezkarnośćplus, #MamyDość, #Kraśnik, #FałszywaPandemia, #plandemia (narrow #plandemia to PL tweets, as it's popular in Spain too)  etc)

*Expected outcome:* a good quality raw dataset, with appropriate scope (see requirements in following tasks)

*Remark:* The dataset in question might be sizeable depending on the choice of a hashtag and a period of time. Consider running the scraping procedure in advance

<br><br>
2. (4 points) 
**Community Detection & Visualisation**:

  * Use the previously obtained data to construct a network of Twitter users participating in the discussion surrounding the controversy. Use user Mentions or Follows as the basis for creating edges within the network.

  * Perform a community detection on obtained network(s), using a selected method available in a Python module of choice. Ensure the communities do *not* overlap

  * Visualise the network(s) and communities obtained in the previous step. Note the size of communities and their placement within the network. What does the obtained structure tell you about controversy in question? 

*Expected outcome:* 
- a set of readable network visualisations, interpretable with the respect to the community structure; 
- thoughts and remarks inferred by the student based on the aforementioned visualisations 

*Remark 1:* depending on the module and its implementation of community detection and the size of your data, the process can take a lengthy amount of time. Consider that when allocating the time for your work.

<br><br>
3. (5 points) 
**Community Analysis** 

Prepare an analysis of the community structure(s) using previously obtained visualisation, full dataset acquired in task 1 and appropriate Python modules.

Extract relevant user-features (eg. usage of other topical hashtags, geolocation, average sentiment, tweeting activity etc) and use them to train a simple classification model, that maps these features to the cluster space (user features -> cluster_id). Review the classification metrics. If you've created two network structures as a part of your work on task 2, review the differences between classification results on each of them.

*Expected outcome:* 
- a thorough analysis of the community structures and meaningful thoughts; 
- a simple classification model trained on the data; 
- inferences and conclusions drawn by the student 

<br><br>
4. (3 points from project proposal pool) 
**Project proposal template fill part II**

Fill next 4 parts of project proposal template (Scope, Data, Timeframe, Key Stakeholders)


<br><br><br><br>
### Readings
[Community detection brief](https://www.analyticsvidhya.com/blog/2020/04/community-detection-graphs-networks/)

[Graph libraries comparison](https://www.timlrx.com/2019/05/05/benchmark-of-popular-graph-network-packages/)
[Graph libraries comparison 2](https://graph-tool.skewed.de/performance)

[Sklearn supervised learning](http://scikit-learn.org/stable/supervised_learning.html)

