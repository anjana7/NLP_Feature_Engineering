#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import tldextract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV



from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()

PATH = Path('C:/Users/anjana1.yadav/Desktop/anjana/Hackerearth_ZS_Associates')
DATA = Path(PATH/'dataset')


# In[ ]:


def missing_percent(train):
    # Calculate percentage of missing keywords
    for c in train.columns.values:
        print(c)
        bool_series_keyword = pd.isnull(train[c]) 
        print('{}% of {} are missing from Total Number of Records\n'.format((len(train[bool_series_keyword])/len(train.index))*100, c))
    
    print("train patient tag counts : ", train.Patient_Tag.value_counts())
    return()


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
    
    filtered_sentence = [w.lower() for w in stripped if (not w in stop_words and w.isalpha() and len(w)>2)]
    
    #lemmatization
    lemma = lemmatize_sentence(' '.join(filtered_sentence))

    return(lemma)

def edit_link(sent):
    punct = list(string.punctuation)
    new_sent = ''
    for s in sent:
        if s in punct:
            new_sent+=' '
        else:
            new_sent+=s
            
    tokens = new_sent.split(" ")
    
    stop_words = stopwords.words('english')
    stop_words.extend(['http','co','com','uk','www','net','html','php','cpp','post','forums','boards','threads','thread','forum','feed','utm','org','page','topic','index','pg','htm','qid','id','default','aspx',''])
    list_to_remove = ['no', 'not']
    
    stop_words = set(stop_words).difference(set(list_to_remove))
    
    filtered_sentence = [w for w in tokens if (not w in stop_words and w.isalpha() and len(w)>3)]
    
    return(filtered_sentence)


def parameter_tuning(X_train, y_train):
    #defining the expected hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(200, 700, num = 11)]
    max_depth.append(None)
    min_samples_split = [10, 20, 50, 80]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print("The set of hyperparameters we are passing to rfc model to tune are:\n", random_grid)

    #Defining the Random Forest Classifier and Randomised search
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1).fit(X_train, y_train)

    print("Optimal Parameters predicted by Randomised Search are: \n", rf_random.best_params_)
    return(rf_random)

def get_domain(url):
    domain = tldextract.extract(url).domain
    return(domain)


# In[ ]:


def get_text_features(df): #df=train

    df.Title = df.Title.fillna(' ')
    df_features = df[['TRANS_CONV_TEXT', 'Title']]
    df_features['Domain'] = df.Link.apply(get_domain)
    df_features.TRANS_CONV_TEXT = df_features.TRANS_CONV_TEXT.progress_apply(preprocess)
    df_features.Title = df_features.Title.progress_apply(preprocess)
    df_features['Link'] = df.Link.apply(edit_link)
    df_features['Link'] = [' '.join(df_features.Link[i] + [df_features.Title[i]]) for i in df_features.index]
    
    return(df_features)

def get_tfidf_vectorizer(df): #df_features
    
    corpus_text = df.TRANS_CONV_TEXT.values
    corpus_link = df.Link.values

    vectorizer_text = TfidfVectorizer(max_df=0.05, max_features=500)
    X_text = vectorizer_text.fit_transform(corpus_text)

    vectorizer_link = TfidfVectorizer(max_df=0.1, max_features=70)
    X_link = vectorizer_link.fit_transform(corpus_link)
    
    return(X_text, X_link, vectorizer_text, vectorizer_link)


def get_label_encode(df1, df2): #df1=train, df2=df_features
    
    df_sub = df1[['Source', 'Link']]

    df_sub.Link = df_sub.Link.apply(get_domain)
    df_sub.Source = df_sub[['Source']].applymap(str.lower)
    
    label_encoder_src = LabelEncoder()
    df_sub['Source'] = label_encoder_src.fit_transform(df_sub.Source).astype('int')

    label_encoder_Link = LabelEncoder()
    df_sub['Link'] = label_encoder_Link.fit_transform(df_sub.Link).astype('int')

    df_sub['word_count'] = df2['TRANS_CONV_TEXT'].apply(lambda x: len(str(x).split(' ')))
    
    return(df_sub, label_encoder_Link, label_encoder_src)

def merge_df(df1, df2, df3): #df1=X_text, df2=X_link, df3=df_sub
    
    df_text = pd.DataFrame(data=csr_matrix(df1).todense())
    df_link = pd.DataFrame(data=csr_matrix(df2).todense())
    df_train = pd.DataFrame(np.hstack([df_text, df_link, df3]))
    
    return(df_train)

def get_train_transforms(train):
    
    df_features = get_text_features(train)
    X_text, X_link, vectorizer_text, vectorizer_link = get_tfidf_vectorizer(df_features)
    df_sub, label_encoder_Link, label_encoder_src = get_label_encode(train, df_features)
    df_final = merge_df(X_text, X_link, df_sub)
    
    return(df_final.values, vectorizer_text, vectorizer_link, label_encoder_Link, label_encoder_src)
    
def get_test_tfidf(df, v_text = vectorizer_text, v_link = vectorizer_link):  #df = df_test_features
    
    X_text = v_text.transform(df.TRANS_CONV_TEXT.values)
    X_link = v_link.transform(df.Title.values)

    return(X_text, X_link)

def encoding_src(sent, label_encoder = label_encoder_src):
    try:
        return label_encoder.transform(sent).astype('int')
    except:
        return 0
    
def encoding_link(sent, label_encoder = label_encoder_Link):
    try:
        return label_encoder.transform(sent).astype('int')
    except:
        return 0

def get_test_encoding(df1, df2): #df1 = test, df2 = df_test_features
    
    df_sub = df1[['Source', 'Link']]
    df_sub.Link = df_sub.Link.apply(get_domain)
    df_sub.Source = df_sub[['Source']].applymap(str.lower)
    df_sub.Source = df_sub.Source.apply(encoding_src)
    df_sub.Link = df_sub.Link.apply(encoding_link)

    df_sub['word_count'] = df2['TRANS_CONV_TEXT'].apply(lambda x: len(str(x).split(' ')))
    df_sub['len_stopwords'] = df1.TRANS_CONV_TEXT.apply(lambda x: len(str(x).split(' ')))
    df_sub.len_stopwords = df_sub.len_stopwords - df_sub.word_count
    
    return(df_sub)

def get_test_transforms(test):
    
    df_test_features = get_text_features(test)
    X_text, X_link = get_test_tfidf(df_test_features)
    df_sub = get_test_encoding(test, df_test_features)
    df_final = merge_df(X_text, X_link, df_sub)
    
    return(df_final.values)


# In[ ]:


# model = parameter_tuning(df_train.values, y)
# rfc = model.best_estimator_
# score_rfc = cross_val_score(rfc, df_train.values, y, cv=3, scoring='f1')

# rfc.fit(X, y)


# In[ ]:


train = pd.read_csv(Path(DATA/'train.csv'), engine='python')
test = pd.read_csv(Path(DATA/'test.csv'), engine='python')

train.drop(['Host', 'Date(ET)','Time(ET)','time(GMT)'], inplace=True, axis=1)
train.dropna(subset=['TRANS_CONV_TEXT'], inplace=True)


# In[ ]:


X, vectorizer_text, vectorizer_link, label_encoder_Link, label_encoder_src = get_train_transforms(train)
y = train.Patient_Tag.values
rfc = RandomForestClassifier(n_estimators=522, min_samples_split=10, min_samples_leaf=1, max_features='auto', max_depth=700, bootstrap=True).fit(X, y)


# In[ ]:


X_test = get_test_transforms(test)
predictions = rfc.predict(X_test)
predictions = np.round(predictions).astype(int).reshape(len(test))

sub=pd.DataFrame({'Index':test['Index'].values.tolist(),'Patient_Tag':predictions})
sub.to_csv(PATH/'submission/submission_features_rfc_improved.csv',index=False)


# In[ ]:




