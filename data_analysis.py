import os
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import random

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
from termcolor import colored

from utils import save, load

class Preprocessor(object):
    """
    Preprocessor
    """
    def __init__(self):
        # Define spacy pipeline
        self.nlp = spacy.load('en_core_web_md')
        self.tfidf = TfidfVectorizer(tokenizer=lambda x: x, min_df = 2, max_df = 0.5, ngram_range = (1, 2), lowercase=False)

    @staticmethod
    def cut_sentences(self, df, max_len=512):
        """
        Fucntion to cut sentence and place it as a new sample

        For example, a sentence with length 1025 will be cut into:
        sentence A with len 512 + sentence B with len 512 + sentence C with len 1
        Other info such as label will be retained

        Is this the best approach? Should we only do it to the negative samples?
        """
        # Should we find the nearest "." symbol to cut the sentences?
        df['paragraph'] = df.paragraph.apply(lambda s: [s[i:i+max_len] for i in range(0, len(s), max_len)])
        df.explode('paragraph', inplace=True)
        return df

    def spacy_preprocess(self, texts, save_path):
        """
        Function to preprocess using spacy tokenizer AND save the resulted tokens
        texts: List of texts
        save_path: path to save
        Return: List of tokens
        """
        print('Preprocessing using spacy pipeline...')
        docs = self.nlp.pipe(texts)
        out_tokens = [self.spacy_tokenizer(doc) for doc in docs]
        save(out_tokens, save_path)
        print('Done!\nPreprocessed token saved at {}'.format(save_path))
        return out_tokens

    @staticmethod 
    def spacy_tokenizer(doc):
        """
        A spacy tokenizer to do preprocessing for Tranditional ML classifier
        """
        out_tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue # Remove stop words
            elif token.like_email:
                out_tokens.append('<email>') # Replayce email 
            elif token.like_url:
                out_tokens.append('<url>') # Replace URL
            elif token.like_num:
                out_tokens.append('<num>') # Replace numbers
            elif token.lemma_ != "-PRON-": # Lemmatisation if not proper noun
                out_tokens.append(token.lemma_.lower().strip()) 
            else:
                out_tokens.append(token.lower_) # Lower case
        return out_tokens

    def get_tfidf_vectors(self, df, train, load_path='spacy_preprocessed', force_rebuild=False):
        """
        Function to build TF-IDF matrix
        df: dataframe to process
        train: bool, if true fit and transform, if false, only transform
        load_path: if present, load tokens from file (No preprocessing of df required)
        force_rebuild: bool, whether to force the re-preprocessing of the df

        Return: TF-IDF matrix, size (#samples, #tokens)
        """
        path = load_path+'_train.pkl' if train else load_path+'_val.pkl'
        if os.path.exists(path) and not force_rebuild:
            print('Loading preprocessed tokens...')
            out_tokens = load(path)
            print('Done!\nPreprocessed token loaded at {}'.format(path))
        else:
            out_tokens = self.spacy_preprocess(df.paragraph, save_path=path)
        
        if train:
            features = self.tfidf.fit_transform(out_tokens)
        else:
            features = self.tfidf.transform(out_tokens)
          
        tfidf_df = pd.DataFrame(
             features.todense(),
             columns = self.tfidf.get_feature_names(),
             index = df.index
        )

        tfidf_df['label'] = df.label
        return tfidf_df


def preprocess(nlp, paragraph):
    tokens_list = []
    sent_list = []
    num = 0
    email = 0
    url = 0
    bracket = 0
    quote = 0
    currency = 0
    oov = 0
    for doc in nlp.pipe(paragraph):
        tokens = defaultdict(list)
        sent_list.append(doc)
        for token in doc:
            tokens['tokens'].append(token)
            if token.like_num:
                tokens['num'].append(token)
                num += 1
            if token.like_email:
                tokens['email'].append(token)
                email += 1
            if token.like_url:
                url +=1
                tokens['url'].append(token)
            if token.is_bracket:
                bracket += 1
            if token.is_quote:
                quote += 1
            if token.is_currency:
                currency += 1
                tokens['currency'].append(token)
            if token.is_oov:
                oov += 1
                tokens['oov'].append(token)
            # if len(tokens) > 3:
            #     tokens.append(doc) 
            # print(token.text, token.like_num, nlp.vocab.strings[token.text])
        # break
        tokens_list.append(tokens)
    return tokens_list, (num,email,url,bracket,quote,currency,oov), sent_list

def highlight_text(words, sents, highlight_type, random=False, num_show='all'):
    if num_show is None or num_show =='all':
        num_show = len(sents)
        random=False
    
    idxs = np.arange(len(sents))
    if random:
        np.random.shuffle(idxs)
    print(idxs)
    for i in idxs:
        if num_show == 0:
            break
        if highlight_type in words[i].keys():
            result = " ".join(colored(t,'white','on_red') if t in words[i][highlight_type] else t.text for t in sents[i])
            print(str(i)+': ', colored(words[i][highlight_type],'white','on_blue'), result+'\n')
            num_show -= 1

def find_data_by_key_words(df, keyword):
    res = df[df.paragraph.str.contains(keyword)]
    return res

# def find_freq_words(df)

if __name__=='__main__':

    from utils import get_df
    path = 'nlp_data'
    df_train, df_test, df_pcl, df_cat = get_df(path)

    pre = Preprocessor()
    X = pre.get_tfidf_vectors(df=df_train, train=True)
    X_test = pre.get_tfidf_vectors(df=df_test, train=False)
    print(X[['<num>','--the','100','1800','label']].describe())


    # Inspect length
    df_train['para_len'] = df_train['paragraph'].apply(len)
    df_train.groupby('label')['para_len'].describe()

    import xgboost as xgb
    d_train = xgb.DMatrix(X[X.columns.drop('label').values], X['label'].values)
    xgb_params = {'eta': 0.05, 
              'max_depth': 12, 
              'subsample': 0.8, 
              'colsample_bytree': 0.75,
              #'min_child_weight' : 1.5,
              'scale_pos_weight': imbalance_weight,
              'objective': 'binary:logistic', 
              'eval_metric': 'auc', 
              'seed': 23,
              'lambda': 1.5,
              'alpha': .6
             }
    # watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 2000)
    # Define spacy pipeline
    # nlp = spacy.load('en_core_web_md')
    # Preprocess
    # pcl_tokens, pcl_stats, pcl_sent_list = preprocess(nlp, df_pcl.paragraph)
    # cat_tokens, cat_stats, cat_sent_list = preprocess(nlp, df_cat.span_text)

    # See what are the oov words
    # highlight_text(cat_tokens, cat_sent_list, 'oov')  






