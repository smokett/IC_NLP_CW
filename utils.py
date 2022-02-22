import os
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc


def get_df(path):
    # Load dataset
    pcl_col_names = ['paragraph_id', 'article_id', 'keyword', 'country_code', 'paragraph','label']
    cat_col_names = ['paragraph_id', 'article_id', 'paragraph', 'keyword', 'country_code', 'span_start', 'span_end', 'span_text', 'category_label', 'number_of_annotators_agreeing_on_that_label']
    df_pcl = pd.read_csv(os.path.join(path, 'dontpatronizeme_pcl.tsv'), sep='\t', skiprows=3, header=None, names=pcl_col_names, index_col='paragraph_id')
    df_cat = pd.read_csv(os.path.join(path, 'dontpatronizeme_categories.tsv'), sep='\t', skiprows=3, header=None, names=cat_col_names)
    
    # df_pcl.dropna(subset=['paragraph'], inplace=True)
    # df_cat.dropna(subset=['paragraph'], inplace=True)
    # 0,1 => No PCL, 2, 3, 4 => PCL
    df_pcl['label'] = 1 * (df_pcl['label'] > 1)

    # Train/test split based on official document
    df_train_index = pd.read_csv(os.path.join(path, 'train_semeval_parids-labels.csv'))
    df_test_index = pd.read_csv(os.path.join(path, 'dev_semeval_parids-labels.csv'))
    df_train = df_pcl.reindex(df_train_index['par_id'])
    df_test = df_pcl.reindex(df_test_index['par_id'])

    df_train.dropna(subset=['paragraph'], inplace=True)
    df_test.dropna(subset=['paragraph'], inplace=True)

    df_train.to_csv('df_train.csv')
    df_test.to_csv('df_test.csv')
    return df_train, df_test, df_pcl, df_cat

def get_ext_df(path):
    ext_col_names = ['paragraph_id', 'article_id', 'keyword', 'country_code', 'paragraph', 'label']
    df_train = pd.read_csv(os.path.join(path, 'df_train_expansion.csv'), names=ext_col_names, index_col='paragraph_id')
    return df_train

def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as f:
        ploter = pickle.load(f)
    return ploter

def evaluate(y_score, y_true):

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(y_score >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(classification_report(y_true, y_pred))
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def cut_sentences(df, df_cat, max_len=512):
        """
        Fucntion to cut sentence and place it as a new sample

        For example, a sentence with length 1025 will be cut into:
        sentence A with len 512 + sentence B with len 512 + sentence C with len 1
        Other info such as label will be retained

        Is this the best approach? Should we only do it to the negative samples?
        """
        # Should we find the nearest "." symbol to cut the sentences?

        def row_update(r):
            num_sent = len(r.paragraph)
            l = [0 for _ in range(num_sent)]
            for i in range(num_sent):
                if isinstance(r.min_start, list):
                    for i in range(len(r.min_start)):
                        s = int(r.min_start[i]//max_len)
                        e = int(r.max_end[i]//max_len) + 1
                        for j in range(s,e):
                            l[j] = 1
            r.label = l
            return r

        def explode(df, col1, col2):
            df['tmp']=df.apply(lambda row: list(zip(row[col1],row[col2])), axis=1) 
            df=df.explode('tmp')
            df[[col1,col2]]=pd.DataFrame(df['tmp'].tolist(), index=df.index)
            df.drop(columns='tmp', inplace=True)
            return df

        min_start = df_cat.groupby('paragraph_id')['span_start'].apply(list)
        max_end = df_cat.groupby('paragraph_id')['span_end'].apply(list)

        
        df['paragraph'] = df.paragraph.apply(lambda s: [s[i:i+max_len] for i in range(0, len(s), max_len)])
        df = df.assign(min_start=min_start, max_end=max_end)
        df = df.apply(row_update,axis=1)
        # For pandas version < 1.3
        df = explode(df, 'paragraph','label')
        # For pandas version > 1.3
        # df = df.explode(['paragraph', 'label']) 
        df.drop(columns=['min_start', 'max_end'], inplace=True)
        return df

if __name__ == '__main__':
    df_train, df_test, df_pcl, df_cat = get_df('nlp_data')
