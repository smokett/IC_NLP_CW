from nltk.corpus import wordnet
import random
import pandas as pd

def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(words, n):
    
    words = words.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

col_names = ['par_id', 'article_id', 'keyword', 'country_code', 'paragraph','label']
df_train = pd.read_csv('df_train.csv', header=0, index_col='par_id')
new_df_train = df_train.copy()
for n in range(1):
    for index, row in df_train.iterrows():
        if row['label'] == 1:
            new_row = row.copy()
            new_row['paragraph'] = synonym_replacement(new_row['paragraph'],10)
            new_df_train = new_df_train.append(new_row, ignore_index=True)

new_df_train.to_csv("synonym_df_train.csv")
