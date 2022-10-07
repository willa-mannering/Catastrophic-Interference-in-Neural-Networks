'''
Experiment 1b - Neural Network in a Natural language corpus (no EWC)

author: @willamannering
date: 10/29/2019

'''

#import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pickle
import collections
import string
import gensim
import os


cwd = os.getcwd()
stopwords = set(stopwords.words("english"))

def fetch_clean_data(data):    
    # split sentences
    sentences = []
    for d in data:
        for sent in d:
            sentences.append(sent)
    # remove '[S]' tag from the sentences
    clean_sentences = ' '
    for s in sentences:
        if '[S]' in s: 
            clean_sentences += ' ' + s[4:]
            
    # remove punctuations
    table = str.maketrans({key: None for key in string.punctuation})
    new_s = clean_sentences.translate(table)    
    
    # remove numbers
    no_numbers = ''.join(i for i in new_s if not i.isdigit())
    
    # remove stop words
    remove_stop = ' '
    tokens = no_numbers.split()
    for n in tokens:
        if n not in stopwords:
            remove_stop += n + ' '
    
    # remove words occuring just once
    cnt = collections.Counter()
    no_stop_tokes = remove_stop.split()
    for word in no_stop_tokes:
        cnt[word] +=1
    
    filtered_dict = {}
    for c in cnt:
        if cnt[c] > 1:
            filtered_dict[c] = cnt[c]
    
    # generate final text for processing
    wor = remove_stop.split()
    text = ' ' 
    for r in wor:
        if r in filtered_dict:
            text += r + ' ' 
    
    data = text.split()
    return data


# seperate sentences 
def seperate_sentences(topic_data):
    sep_sent = []
    for d in range(0, len(topic_data)-11):
        sep_sent.append(topic_data[d : d+11])
    return sep_sent


# import topic 1 sense (dominant), from text file
fname1 = open(cwd + '/science.txt', 'rb')
file1 = pickle.load(fname1)

# import topic 2 sense (subordinate), from text file
fname2 = open(cwd + '/business.txt', 'rb')
file2 = pickle.load(fname2)

# collect and clean topic 1 sentences
topic1_sent = seperate_sentences(fetch_clean_data(file1))
# collect and clean topic 2 sentences
topic2_sent = seperate_sentences(fetch_clean_data(file2))

#Random ordering
combine = topic2_sent + topic1_sent
np.random.shuffle(combine)

# topic 1 then topic 2 ordering
#combine = topic1_sent + topic2_sent

# topic 2 then topic 1 ordering
#combine = topic2_sent + topic1_sent

# create similairty dictionary
sim_dict = defaultdict(dict)

# target word (homonym)
target_word = 'slip'

# comparison words (words synonymous to target work in two topics)
compare_words= ['buckles', 'cashiers']

# initialize similarity dictionary with empty lists 
sim_dict[target_word][compare_words[0]] = []
sim_dict[target_word][compare_words[1]] = []

number_of_runs = 200
# sg - skipgram
# alpha - learning rate
# size - embedding size
# window - context window
# seed - random number seed
# min_count - words less than min_count are disgarded
# negative - number of negative samples drawn
# iter - number of iterations
for i in range(0, number_of_runs):
    print (i)
    model = gensim.models.word2vec.Word2Vec(combine, sg = 1, 
                                            alpha = 0.16,
                                            size = 50,
                                            window = 5,
                                            seed = i,
                                            min_count = 0,
                                            hs = 0,
                                            negative = 1,
                                            iter = 1)   

# save similarity between target and comparison word one
    sim_dict[target_word][compare_words[0]].append(cosine_similarity([model.wv[target_word]], [model.wv[compare_words[0]]])[0][0])
# save similarity between target and comparison word two
    sim_dict[target_word][compare_words[1]].append(cosine_similarity([model.wv[target_word]], [model.wv[compare_words[1]]])[0][0])


print ('Mean')
print (target_word + ' - ' + compare_words[0] + ' :', np.mean(sim_dict[target_word][compare_words[0]])) 
print (target_word + ' - ' + compare_words[1] + ' :', np.mean(sim_dict[target_word][compare_words[1]])) 
print ('Standard Deviation')
print (target_word + ' - ' + compare_words[0] + ' :', np.std(sim_dict[target_word][compare_words[0]])) 
print (target_word + ' - ' + compare_words[1] + ' :', np.std(sim_dict[target_word][compare_words[1]])) 

#dframe = pd.DataFrame(sim_dict)
#dframe.to_pickle('<name_to_save>.pkl')
    
    
