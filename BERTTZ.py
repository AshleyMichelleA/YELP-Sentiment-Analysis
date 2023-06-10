#!/usr/bin/env python
# coding: utf-8

# INSTALL AND IMPORT

get_ipython().system('pip install torch torchvision torchaudio')


# EXPLANATION OF EACH LIBRARY BELOW
# transformers: To leverage transformers to our NLP model, easily import and install using BERT
# 
# requests: Makes a request to the site.
# 
# beautifulsoup4: Extract the data.
# 
# pandas: Structure our data.
# 
# numpy: Additional data transformation processes.

# In[1]:


get_ipython().system('pip install transformers requests beautifulsoup4 pandas numpy')


# DEPENDENCIES EXPLANATION:
# 
# AutoModelForSequenceClassification:  
# The tokenizer is going to allow us to pass through a string and convert that into a sequence of numbers that we can pass to out NLP model
# 
# AutoModelForSequenceClassification: 
# Its going to give us the architecture from transformers to be able to load in the NLP model.
# 
# torch : Imported pyTorch this is to use the arg max function to extract the highest sequence result.
# 
# requests: Grab Data
# 
# BeautifulSoup: Extract ut
# 
# re: Creates a regent instruction to extract the specific coments

# In[2]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
import requests
from bs4 import BeautifulSoup
import re


# In[3]:


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[4]:


tokens = tokenizer.encode('I love this', return_tensors='pt')


# In[5]:


result = model(tokens)


# RESULTS EXPLANATION : 
# Number index from 1-5 index with highest values is the score of the comment

# In[6]:


result.logits


# In[7]:


int(torch.argmax(result.logits))+1


# In[8]:


r = requests.get('https://www.yelp.com/biz/taqueria-los-gallos-express-concord-3?osq=Taqueria+Los+Gallos')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})
reviews = [result.text for result in results]


# In[9]:


reviews


# In[10]:


import numpy as np
import pandas as pd


# In[11]:


df = pd.DataFrame(np.array(reviews), columns=['review'])


# In[12]:


df['review'].iloc[0]


# In[13]:


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# In[14]:


sentiment_score(df['review'].iloc[1])


# In[15]:


df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))


# In[16]:


df


# In[18]:


df['review'].iloc[8]







