'''
Experiment 1a - CI in an artificial language (no EWC)

author: @willamannering
date: 10/29/2019

'''

import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# number of repetitions
n_sent_reps = 1000

# fish sense
fish_data = ['bass fish',
          'bass eat', 
          'trout fish',
          'trout eat',]

# instrument sense
instrument_data = ['bass play',
          'bass pluck',
          'acoustic play',
          'acoustic pluck']


# generates dominant and subordinate repetitions
def generate_dom_sub(data, dominant, reps):
    inner_list = []
    if dominant == True:
        for d in data:
            inner_list += [d] * reps
    if dominant == False:
        for d in data:
            inner_list += [d] * (int(reps/3))
    return inner_list

# if dominant = True, the data is considered a dominant sense
# if dominant = False, the data is considered a subordinate sense
fish = generate_dom_sub(fish_data, dominant = True, reps = n_sent_reps)
instrument = generate_dom_sub(instrument_data, dominant = True, reps = n_sent_reps)

# convert words into indexes
vocab = ['bass', 'acoustic', 'trout', 'fish', 'eat', 'play', 'pluck']
word_to_index, index_to_word = {}, {}
for v in range(0, len(vocab)):
    word_to_index[vocab[v]] = v
    index_to_word[v] = vocab[v]
    
# create a 2d-dictionary with list
sim_dict = defaultdict(dict)

# initialize the similarity dictionary
for v in vocab:
    for ve in vocab:
        sim_dict[v][ve] = []

# number of runs
num_runs = 200

# embedding size
embedding_size = 10

# for each run
for i in range(0, num_runs):
    
    # shuffle each sense
    np.random.shuffle(fish)
    np.random.shuffle(instrument)
    
    # uncomment for random case
    generate_corpus = instrument + fish
    np.random.shuffle(generate_corpus)
    
    # uncomment for FI case
    #generate_corpus = instrument + fish

    # uncomment for IF case
    #generate_corpus = fish + instrument


    # reset graph at every run
    print (i)
    tf.reset_default_graph()
    
    # embedding matrix
    embeddings = tf.Variable(tf.random_uniform([len(vocab), embedding_size], -1.0, 1.0, seed = i))
    
    # weight matrix
    nce_weights = tf.Variable(tf.truncated_normal([len(vocab), embedding_size], stddev = 1.0/np.sqrt(embedding_size), seed = i))
    nce_biases = tf.Variable(tf.zeros([len(vocab)]))
    
    # input and output placeholders
    train_inputs = tf.placeholder(tf.int32, shape = [1]) #shape = [batchsize]
    train_labels = tf.placeholder(tf.int32, shape = [1, 1])

    # collect embeddings    
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
    # initialize loss function (https://www.tensorflow.org/tutorials/word2vec)
    loss = tf.reduce_mean(
            tf.nn.nce_loss(weights= nce_weights,
                           biases= nce_biases,
                           labels = train_labels,
                           inputs = embed,
                           num_sampled= 1,
                           num_classes= len(vocab))
            )
    
   
    # initialize optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.02).minimize(loss)

    
    # create input and output feeds
    input_feed = []
    output_feed = []
    for c in generate_corpus:
        splitted = c.split()
        input_feed.append(word_to_index[splitted[0]])
        output_feed.append(word_to_index[splitted[1]])
    
    # reshape the lists
    input_feed = np.array(input_feed)
    output_feed = np.reshape(np.array(output_feed), (len(output_feed), 1))
    
    # initialize tensorflow session
    sess = tf.Session()
    
    # initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # update weights after example
    for e in range(0, len(input_feed)): #for all items in input
        x = sess.run(optimizer, feed_dict = {train_inputs: [input_feed[e]], train_labels: [output_feed[e]]})


    # collect vectors 
    inp_vectors = {}
    for v in range(0, len(vocab)):
        inp_vectors[vocab[v]] = sess.run(embed, feed_dict={train_inputs: [word_to_index[vocab[v]]]})

    # calculate similarities
    for v in inp_vectors:
        for vv in inp_vectors:
            sim_dict[v][vv].append(cosine_similarity(inp_vectors[v], inp_vectors[vv])[0][0])
            

print ('Bass - Acoustic: ', np.mean(sim_dict['bass']['acoustic']))
print ('Bass - Trout: ', np.mean(sim_dict['bass']['trout']))

print ('Bass - Acoustic Std: ', np.std(sim_dict['bass']['acoustic']))
print ('Bass - Trout Std: ', np.std(sim_dict['bass']['trout']))

#dframe = pd.DataFrame(sim_dict)
#dframe.to_pickle('Random_Isub_100runs.pkl')

sess.close()