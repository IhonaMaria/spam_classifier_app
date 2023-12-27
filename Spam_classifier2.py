#!/usr/bin/env python
# coding: utf-8

# # SPAM CLASSIFIER

# ## 1. Data access

# In[1]:


import os

DATASETS_DIR = 'datasets'
MODELS_DIR = 'models'
TAR_DIR = os.path.join(DATASETS_DIR, 'tar')

SPAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'
EASY_HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2'
HARD_HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2'


# In[2]:


# Download and extract the contents of a tar archive from the URL

from urllib.request import urlretrieve
import tarfile
import shutil

def download_dataset(url):
    """download and unzip data from a url into the specified path"""
    
    # create directory if it doesn't exist
    if not os.path.isdir(TAR_DIR):
        os.makedirs(TAR_DIR)
    
    filename = url.rsplit('/', 1)[-1]
    tarpath = os.path.join(TAR_DIR, filename)
    
    # download the tar file if it doesn't exist
    try:
        tarfile.open(tarpath)
    except:
        urlretrieve(url, tarpath)
    
    with tarfile.open(tarpath) as tar:
        dirname = os.path.join(DATASETS_DIR, tar.getnames()[0])
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)
        tar.extractall(path=DATASETS_DIR)
        
        cmds_path = os.path.join(dirname, 'cmds')
        if os.path.isfile(cmds_path):
            os.remove(cmds_path)
    
    return dirname


# In[3]:


# download the data
spam_dir = download_dataset(SPAM_URL)
easy_ham_dir = download_dataset(EASY_HAM_URL)
hard_ham_dir = download_dataset(HARD_HAM_URL)


# In[4]:


import numpy as np
import glob

def load_dataset(dirpath):
    """load emails from the specified directory"""
    
    files = []
    filepaths = glob.glob(dirpath + '/*')
    for path in filepaths:
        with open(path, 'rb') as f:
            byte_content = f.read()
            str_content = byte_content.decode('utf-8', errors='ignore')
            files.append(str_content)
    return files
    


# In[5]:


# load the datasets
spam = load_dataset(spam_dir)
easy_ham = load_dataset(easy_ham_dir)
hard_ham = load_dataset(hard_ham_dir)


# In[6]:


import sklearn.utils

# create the full dataset
X = spam + easy_ham + hard_ham
y = np.concatenate((np.ones(len(spam)), np.zeros(len(easy_ham) + len(hard_ham))))  # 1 for spam, 0 for non-spam

# shuffle the dataset
X, y = sklearn.utils.shuffle(X, y, random_state=42)


# In[7]:


from sklearn.model_selection import train_test_split

# split the data into stratified training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                    random_state=42)



# In[8]:




# ## 2. Data preparation

# - Remove Headers: You might want to remove or selectively process email headers, as they can contain a lot of technical data that may not be useful for your analysis.
# - Handle HTML Content: If the email body contains HTML, strip out the HTML tags to leave just the text.
# - Lowercasing: Convert all text to lowercase to ensure uniformity, as case usually doesn't affect spamminess.
# - Remove Special Characters and Numbers: You might want to remove numbers and special characters, depending on your focus. For spam detection, some characters or number patterns could be relevant, so this step should be considered carefully.
# - Tokenization: Break the text into individual words or tokens

# #### Removing the headers
# 
# Email headers are usually followed by a double newline (\n\n), separating them from the body. We can split the text at the first occurrence of \n\n and take the part after it as the email body.

# In[9]:


def remove_email_headers(email_text):
    """
    Removes headers from an email.
    Assumes headers and body are separated by a \n\n
    """
    parts = email_text.split('\n\n', 1)  # one split only
    if len(parts) > 1:   #Everything after the \n\n is the body
        return parts[1]
    else:
        return email_text  # Return original if no headers found





# #### Handle HTML content
# 
# Email headers are usually followed by a double newline (\n\n), separating them from the body. We can split the text at the first occurrence of \n\n and take the part after it as the email body.

# In[10]:


from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)
    
    
def strip_tags(html):   # This function removes the HTML tags
    s = MLStripper()
    s.feed(html)
    return s.get_data()






# #### Text homogenization
# 
# - First, we'll create a function to convert the URL directions into just 'URL'.
# - Then, all the numbers we'll be transformed into 'NUM'. This can be useful to normalize or reduce the variability of the text.
# - The last function removes all the punctuation signs and returns everything in lowecase to ensure uniformity.
# 

# In[11]:


import re 

def is_url(s):
    url = re.match("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
                     "[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", s)
    return url is not None

def convert_url_to_word(words):
    """convert all urls in the list to the word 'URL'"""
    for i, word in enumerate(words):
        if is_url(word):
            words[i] = 'URL'
    return words


# In[12]:


def convert_num_to_word(words):
    """convert all numbers in the list to the word 'NUM'"""
    for i, word in enumerate(words):
        if word.isdigit():
            words[i] = 'NUM'
    return words


# In[13]:


import nltk
#nltk.download('punkt')
#nltk.download('stopwords')


# #### Using NLTK library
# 
# NLTK is a library that enables us to preprocess text data for further analysis. Tokenization is the process of breaking down text into smaller units, typically words or phrases. In this function, tokenization is achieved by first removing punctuation and replacing tabs and newlines with spaces, and then splitting the text into individual words (or "tokens") based on spaces.
# The function below performs several steps:
# 
#     1. Remove Punctuation: Removes all punctuation from the text to focus only on words
#     2. Replace Tabs and New Lines: Converts tabs (\t) and new lines (\n) to spaces for consistent whitespace handling.
#     3. Split into Words and Remove Stopwords: Splits the text into words and filters out any empty tokens and words that are in the list of stopwords.
#     4. Apply Stemming: Each word is then stemmed using the Porter Stemmer. Stemming is the process of reducing words to their base or root form. For example, "running", "runs", and "ran" are all reduced to the stem "run". 
#     5. Return Processed Tokens: The function returns a list of these processed and stemmed tokens.

# In[14]:


import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

def tokenize_email(text):
    """
    Tokenize a text string. This involves removing punctuation, 
    replacing tabs and newlines with spaces, splitting into words, 
    and then applying stemming. It also filters out stopwords.

    :param text: The string to be tokenized.
    :return: A list of stemmed tokens.
    """
    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Replace tabs and new lines with spaces
    text = text.replace("\t", " ").replace("\n", " ")

    # Split into words and filter out empty tokens and stopwords
    tokens = [word for word in text.split(" ") if word and word not in stop_words]

    # Apply stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    return stemmed_tokens




# #### CUSTOM TRANSFORMER AND PIPELINE
# 
# Let's create a custom transformer and pipeline to apply all the processing steps in order.

# In[15]:


from sklearn.base import BaseEstimator, TransformerMixin

class ProcessEmail(BaseEstimator, TransformerMixin):
    def __init__(self, header=True, strip=True, url_to_word=True, num_to_word=True,
                 remove_punc=True, tokenize_email=True,):
        self.header = header
        self.strip = strip
        self.url_to_word = url_to_word
        self.num_to_word = num_to_word
        self.remove_punc = remove_punc
        self.tokenize_email=tokenize_email
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_cleaned = []
        for email in X:
            if self.header:
                email = remove_email_headers(email)
                
            if self.strip:
                email_stripped=strip_tags(email)
            
            if self.tokenize_email:
                email_tokenized = tokenize_email(email_stripped)
                
            if self.url_to_word:
                email_words = convert_url_to_word(email_tokenized)   
                
            if self.num_to_word:
                email_words = convert_num_to_word(email_words)
            email = ' '.join(email_words)
            
            if self.remove_punc:
                email = email.lower()



            X_cleaned.append(email)
            
        return X_cleaned


# In[16]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# full preparation pipeline
prepare_pipeline = Pipeline([
    ('clean_email', ProcessEmail()),
    ('bag_of_words', CountVectorizer())
])

# CountVectorizer converts a collection of tokenized text documents into a matrix of token counts.


# In[17]:


prepare_pipeline


# In[18]:


X_train_prepared = prepare_pipeline.fit_transform(X_train)







