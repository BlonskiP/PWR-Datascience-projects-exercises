# Assignment 4 (mostly individual)
## Analysis of users emotions in texts

### Scope: knowledge and skils
* understanding Twitter data
* Twitter data acquisition
* basic textual analysis
* Jupiter ipython notebooks (or Jupiter Lab)
* python:  
    * data organization: pandas
    * machine learning: sckikit-learn, xgboost, Keras, PyTorch, Tensorflow
    * NLP: HuggingFace
    * scrapping: scrapy, twint
* teamwork

<br><br><br><br>

### Tasks

1. (4 points) **Data Representation**

A. Use provided dataset (acquired from [DeepMoji](https://github.com/bfelbo/DeepMoji)) project and use 80%/20% train/test split (use scikit functions).   
Labels are arranged in the following way: [joy, fear, anger, sadness, disgust, shame, guilt]  

B. Use some sentence embedding to create sentence representations.

While you're free to use any existing embedding (here is a list of [awesome-sentence-embedding](https://github.com/Separius/awesome-sentence-embedding) and [awesome-2vec](https://github.com/MaxwellRebo/awesome-2vec])), it's highly recommended to use some English-pretrained models wrapped by HuggingFace. We suggest using one of this:
* [DistillBERT](https://huggingface.co/transformers/model_doc/distilbert.html)
* [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)
* [XLM](https://huggingface.co/transformers/model_doc/xlm.html)
* [XLNet](https://huggingface.co/transformers/model_doc/xlnet.html)

*Expected outcome:* Representation of sentences that can be used to feed the neural network 

<br><br>

2. (4 points) **Classifier building**. Build an MLP network using any deep learning framework, eg. Keras, Tensorflow, PyTorch to classify sentences

*Expected outcome:* Neural Network Model that can predict Emotions based on the text. 

*Remark:* You can use the architecture that you prepared for Deep Learning Course. If you find the result unsatisfactory, you may consider mode compound classification method.

<br><br>

3. (3 points) **Validation**. Scrape a few hundreds English Tweets about some emotional topic (eg. BLM, Trump-Biden debate or elections related) and, annotate them and use to validate the trained model. Analyze obtained predictions

*Expected outcome:* Comprehensive analysis of predictions for acquired data

*Remark:* It's not a problem at all to do some group work and split the annotation work through more people (eg. your project partners). If doing so, you can follow standard annotation rules and use a 2+1 model (each text is annotated by 2 annotators, if their scores differ, third one is reviewing this example) 

<br><br>

4. *REMEMBER TO FILL THE WHOLE PROJECT PROPOSAL FORM!*

<br><br><br><br>

### Readings
[Using HuggingFace models for sentence classification](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)  
[Training an embedding in Tensorflow](https://www.tensorflow.org/guide/embedding)  
