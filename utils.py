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
    return df_train, df_test, df_pcl, df_cat

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

