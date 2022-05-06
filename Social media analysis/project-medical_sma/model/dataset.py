import numpy as np
import re
import demoji

demoji.download_codes()

def load_embeddings(filepath):
    loaded = np.load(filepath)
    X = loaded['embeddings']
    y = loaded['labels']
    label_mappings = loaded['label_mappings']
    return X, y, label_mappings

def purify_text(text):
    text = rm_special_tokens(text)
    text = rm_small_words(text)
    return text 

def rm_special_tokens(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls = re.findall(regex,text) 
    emoji = demoji.findall(text)
    for url in urls:
        text = text.replace(url[0],'')
    for emo in emoji.keys():
        text = text.replace(emo, '')
    regex_mentions = r"@\w+"
    mentions = re.findall(regex_mentions,text)
    for men in mentions:
        text = text.replace(men,'')
    trash_symbols = ["#",'\n']
    for tags in trash_symbols:
        text = text.replace(tags,'')
    return text

def rm_small_words(text):
    SMALL_WORDS = ['czy','za','na','u','i','lub','bo','ale','co','o','po','z']
    for word in SMALL_WORDS:
        text=text.replace(word.lower(),'')
        text=text.replace(word.upper(),'')
        text=text.replace(word.casefold(),'')
        text=text.replace(word.capitalize(),'')
        text=text.replace(word,'')
    return text
