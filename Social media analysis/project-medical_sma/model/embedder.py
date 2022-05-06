from transformers import AutoTokenizer, AutoModel, BertForMaskedLM
import torch
import numpy as np
from tqdm import tqdm
import gc

HERBERT = "allegro/herbert-base-cased"

class Embedder:
    def __init__(self, model=HERBERT, cuda=True, batch=34, word_embeddings=False, max_length=200):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = BertForMaskedLM.from_pretrained(model, return_dict=True)
        self.word_mode = word_embeddings
        self.batch = batch 
        self.cuda = cuda
        self.max_length = max_length
    
    """
    Make embeddings
    :param texts: list of texts to embed
    """
    def make_embeddings(self, texts):
        inputs = self.tokenizer(texts,
            add_special_tokens=not self.word_mode,
            padding = 'max_length',
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
            )
        print("Tokenization completed!")

        if self.cuda:
            # cudify model
            self.model.to('cuda')
            inputs.to('cuda')

        step = self.batch
        embeddings = []
        for i in tqdm(range(0, len(texts), step)):
            input_ids = inputs.input_ids[i:i+step]
            attention_mask = inputs.attention_mask[i:i+step]
        
            # calc sentence embeddings

            outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            word_embeddings = outputs.hidden_states[-1]
            if self.word_mode:
                embeddings.append(word_embeddings.cpu().detach().numpy())
            else:
                sentence_embedding = word_embeddings[:,0,:].cpu().detach().numpy()
                embeddings.append(sentence_embedding)
        
            # cleanup
            sentence_embedding = None
            word_embeddings = None
            outputs = None 
            input_ids = None
            attention_mask = None
            if self.cuda:
                torch.cuda.empty_cache()
            gc.collect()
    
        # stack them
        all_sentence_embeddings = np.vstack(embeddings)
        
        return all_sentence_embeddings
