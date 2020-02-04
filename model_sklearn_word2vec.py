#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import string
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from spellchecker import SpellChecker
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import RandomizedSearchCV
from gensim.models import Word2Vec

from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()

PATH = Path('C:/Users/anjana1.yadav/Desktop/anjana/Hackerearth_ZS_Associates')
DATA = Path(PATH/'dataset')

MAX_LEN=100


# In[2]:


def missing_percent(train):
    # Calculate percentage of missing keywords
    for c in train.columns.values:
        print(c)
        bool_series_keyword = pd.isnull(train[c]) 
        print('{}% of {} are missing from Total Number of Records\n'.format((len(train[bool_series_keyword])/len(train.index))*100, c))
    
    print("train patient tag counts : ", train.Patient_Tag.value_counts())
    return()


# In[91]:


def decontracted(phrase):
    '''
    words in short format needs to be corrected. eg: won't = will not.
    If not done then after preprocessing singular words like "t" will
    remain and give no meaning.
    '''
    
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def nltk_tag_to_wordnet_tag(nltk_tag):
    '''
    The pos-tag format given by the nltk tag does not match the pos-tag 
    format of wordnet. So we need to change the formats. 
    '''
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
def lemmatize_sentence(sentence):
    
    '''
    The lemmatizer works better when the pos-tag is also given.
    Lemmatize sentence calculates the pos tag of each word and then
    lemmatizes it.
    '''
    
    lemmatizer = WordNetLemmatizer()
    
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return (" ".join(lemmatized_sentence))


def remove_emoji(text):
    
    '''
    Remove the emojis that might be present in the twitter texts
    '''
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess(sent):
    
    '''
    Preprocess each sentence by following steps:
    Decontraction -> Punctuation removal -> 
    Stopword removal -> Lemmatization
    '''
    
    #de contract the sentence eg. : don't = do not
    decont_sent = decontracted(remove_emoji(sent))
    word_tokens = word_tokenize(decont_sent)
    
    #remove punctuations
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in word_tokens]
    
    #remove stopwords
    stop_words = stopwords.words('english')
    list_to_remove = ['no', 'not']
    stop_words = set(stop_words).difference(set(list_to_remove))
#     spell = SpellChecker()
    
    filtered_sentence = [w for w in stripped if (not w in stop_words and w.isalpha())]
    
    #lemmatization
    lemma = lemmatize_sentence(' '.join(filtered_sentence))

    return(lemma)

def fit_word2_vec(corpus):
    
    model = Word2Vec(min_count=1, size=10)
    model.build_vocab(corpus)  # prepare the model vocabulary
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
    word_vectors = model.wv
    
    return(word_vectors, model)


# In[4]:


train = pd.read_csv(Path(DATA/'train.csv'), engine='python')
test = pd.read_csv(Path(DATA/'test.csv'), engine='python')


# In[99]:


df_features.TRANS_CONV_TEXT = df_features.TRANS_CONV_TEXT.progress_apply(preprocess)


# In[117]:


from sklearn.feature_extraction.text import TfidfVectorizer

def create_corpus(df):
    '''
    creating a corpus consisting of all the words present 
    in the text.
    '''
    
    '''
    get the most important words in the text documents using tf-idf
    '''
    corpus_row = df.TRANS_CONV_TEXT.values
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=70)
    vectorizer.fit_transform(corpus_row)
    imp_words = vectorizer.get_feature_names()
    
    text = df.TRANS_CONV_TEXT.values
    corpus=[]
    for tweet in tqdm_notebook(text):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==True) and (word in imp_words))]
        corpus.append(list(set(words)))
    return corpus

corpus = create_corpus(df_features)

word_vectors, model = fit_word2_vec(corpus)


# In[223]:


def create_word_embeddings(sent, word_vec=word_vectors, model=model):
    '''
    Word2vec model trained words are taken as word embeddings.
    '''
    
    MAX_WORDS = 30
    embed=[]
    text = sent.split(" ")
    
    w = list(filter(lambda x: x in model.wv.vocab, text))
    
    for word in w:
        embed.append(np.asarray(word_vec[word]))
    
    if len(embed) < MAX_WORDS:
        embed.extend([[0]*10]*(MAX_WORDS-len(embed)))
    elif len(embed) > MAX_WORDS:
        embed[:MAX_WORDS]
    
    return(np.ravel(np.mean(np.asarray(embed), axis=0)))

df_features['TRANS_CONV_TEXT_embed'] = df_features.TRANS_CONV_TEXT.progress_apply(create_word_embeddings)
df_features['Title'] = train[['Title']]
df_features.head()


# In[224]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X_train = df_features.TRANS_CONV_TEXT_embed.values
y_train = df_features.Patient_Tag.values

rfc= RandomForestClassifier(n_estimators=1577, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=70, bootstrap=False)
score_rfc = cross_val_score(rfc, list(X_train), y_train, cv=3, scoring='f1')


# In[226]:


rfc.fit(list(X_train), y_train)


# In[227]:


X_test = list(test.TRANS_CONV_TEXT.progress_apply(create_word_embeddings).values)


# In[231]:


predictions = rfc.predict(X_test)
predictions = np.round(predictions).astype(int).reshape(len(X_test))

sub=pd.DataFrame({'Index':test['Index'].values.tolist(),'Patient_Tag':predictions})
sub.to_csv(PATH/'submission/submission_wrd2vec_rfc.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




