import pandas as pd #Dependencies for data preprocessing
import numpy as np
import nltk
nltk.download('punkt_tab')
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

import unidecode
import re 

stopwords={'before', 'when', 'again', 'how', 'm', 'are', 'yourselves', 'why', "should've", 'once', 'the', 'wouldn', 'is', "aren't", 'of', 'on',
           'your', 'some', 'so', 'too', 'then', 'our', 'own', 'do', 'me', "you've", 'and', "hasn't", "you're", 'other', 'we', "mightn't", "shouldn't",
           's', 'this', 'ours', 'these', 'further', 'above', 'for', "you'd", 'who', 'o', 'that', 'down', 'aren', 'y', 'd', 'where', 'ain', 'have', "shan't",
           'nor', 'about', 'whom', 'not', 'up', 'but', 'am', 'weren', 'mustn', 'through', "hadn't", 'few', 'out', 'to', 'was', "needn't", 'those', "don't", 
           'him', 'with', 'ma', 'will', 'while', 'isn', 'needn', 'each', 'himself', 're', 'there', 'haven', 'off', 've', 'an', 'does', 'should', 'been', 'you', 
           'be', 'it', 'same', 'were', 'she', 'they', 'a', 'over', 'what', 'more', 'itself', "she's", 'which', 'most', 'its', 'no', 'hers', "didn't", 'his', 'don', 
           "that'll", 'only', 'into', 'themselves', 'all', 'couldn', 'by', 'under', 'between', "wasn't", 'won', 'herself', 'because', 'at', 'yours', 'such', 'i', 
           'theirs', "doesn't", 't', 'hasn', "isn't", 'here', 'myself', 'as', 'them', 'ourselves', 'can', 'had', 'if', 'their', "wouldn't", 'just', "it's", "won't", 
           'or', 'mightn', 'from', 'below', "haven't", 'very', 'any', 'll', 'shouldn', 'during', 'against', 'doesn', 'doing', 'now', 'being', 'wasn', 'after', "mustn't",
           'did', 'in', 'he', 'shan', 'having', 'both', 'has', 'my', 'than', 'hadn', "weren't", 'didn', 'until', "couldn't", 'yourself', 'her', "you'll"
           }


def text_handle(col,stopwords=stopwords):
 

  # Convert to lowercase
  text = col.lower()

  # Expand Contractions
  text = contractions.fix(text)

  # Remove URLs
  text = re.sub(r'http\S+', '', text)

  # Remove numbers
  text = re.sub(r'\d+', '', text)

  # Remove punctuation
  text = re.sub(r'[^\w\s.]', '', text)

  # Remove extra spaces
  text = re.sub(r'\s+', ' ', text).strip()

  # Remove accents
  text = unidecode.unidecode(text)
  

  # Remove stopwords
  text = ' '.join([word for word in text.split() if word not in stopwords])

  # Tokenize words
  words = word_tokenize(text)

  # Stem words
  stemmer = SnowballStemmer('english')
  words = [stemmer.stem(word) for word in words]

  # Join words back into a single string
  text = ' '.join(words)

  return text
 
 


#ans=text_handle("CUMBERLAND FARMS 168MANORVILLE NY")
#print(ans)
