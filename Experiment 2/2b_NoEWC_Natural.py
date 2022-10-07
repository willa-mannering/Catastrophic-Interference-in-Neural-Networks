'''
Experiment 2b - Neural Network in a Natural language corpus

author: @willamannering
date: 10/29/2019

'''

#################### Set up Vocab + functions ##############################3

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
    
    # collect unique vocabulary
    vocab = np.unique(remove_stop.split())
    
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
    return data,vocab

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
   
# seperate sentences with a window of size 5
def seperate_sentences(topic_data):
    sep_sent = []
    for d in range(0, len(topic_data)-5):
        sep_sent.append(' '.join(topic_data[d : d+5]))
    return sep_sent

def get_vocab(words):
    v = []
    for w in words:
        v.append(w.split())
    return np.unique(v)


def read_data(file1path, file2path):
# topic 1 sense (dominant)
    fname1 = open(file1path, 'rb')
    file1 = pickle.load(fname1)
    
    # topic 2 sense (subordinate)
    fname2 = open(file2path, 'rb')
    file2 = pickle.load(fname2)
    
    return file1, file2


#Train NN in random order
def trainrand(t1, t2, wti, itw, vocab_len, runs, embed_size, lr, negsamps, target, sense1, sense2):
    
    print('training random...')
    # number of runs
    num_runs = runs
    
    # embedding size
    embedding_size = embed_size
    
    #create smaller, target vocab
    target_vocab = [target,sense1,sense2]
    
    #initialize array to determine average time per run
    avg_time = []
    
     # create a 2d-dictionary with list
    sim_dict = defaultdict(dict)
    
    # initialize the similarity dictionary
    for v in target_vocab:
        for ve in target_vocab:
            sim_dict[v][ve] = []
            
    
    #prepare data
    d = t1+t2
    np.random.shuffle(d)
    data1 = split_list(d)[0]
    data2 = split_list(d)[1]
  
    
    #create feed for task 1
    input_feed_task1 = []
    output_feed_task1 = []
    

    for c in data1:
        splitted = c.split()
        input_feed_task1.append(wti[splitted[0]])
        output_feed_task1.append(wti[splitted[1]])
    
    # reshape the lists
    input_feed_task1 = np.array(input_feed_task1)
    output_feed_task1 = np.reshape(np.array(output_feed_task1), (len(output_feed_task1), 1))
    

    
    #create feed for task 2
    input_feed_task2 = []
    output_feed_task2 = []
    for c in data2:
        splitted = c.split()
        input_feed_task2.append(wti[splitted[0]])
        output_feed_task2.append(wti[splitted[1]])
    
    # reshape the lists
    input_feed_task2 = np.array(input_feed_task2)
    output_feed_task2 = np.reshape(np.array(output_feed_task2), (len(output_feed_task2), 1))


    #initialize array to determine average time per run
    avg_time = []
    
    # for each run
    for i in range(0, num_runs):
        start_time = time.time()

        
        
        #reet tensorflow graph
        tf.reset_default_graph()

        # input and output placeholders
        train_inputs = tf.placeholder(tf.int32, shape = [1]) #shape = [batchsize]
        train_labels = tf.placeholder(tf.int32, shape = [1, 1])
    
        #create embedding matrix
        embeddings = tf.Variable(tf.random_uniform([vocab_len, embedding_size], -1.0, 1.0))
        
        # collect embeddings    
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        # weight and bias matrix
        weights = tf.Variable(tf.truncated_normal([vocab_len, embedding_size], stddev = 1.0/np.sqrt(embedding_size), seed = i))
        biases = tf.Variable(tf.zeros([vocab_len]))
 
        #define loss function: nce
        loss = tf.reduce_mean(
        tf.nn.nce_loss(weights= weights,
                       biases= biases,
                       labels = train_labels,
                       inputs = embed,
                       num_sampled= negsamps,
                       num_classes= vocab_len)
        )
        
        #define optimizer: gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)


        # initialize tensorflow session
        sess = tf.Session()
        
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        
        #RUN TASK 1
        
        # update weights after example
        for e in range(0, len(input_feed_task1)): #for all items in input
            x = sess.run([loss, optimizer], feed_dict = {train_inputs: [input_feed_task1[e]], train_labels: [output_feed_task1[e]]})
            
        #RUN TASK 2
        # update weights after example
        for e in range(0, len(input_feed_task2)): #for all items in input
            x = sess.run([loss, optimizer], feed_dict = {train_inputs: [input_feed_task2[e]], train_labels: [output_feed_task2[e]]})
        
        # collect vectors 
        inp_vectors = {}
        for v in range(0, len(target_vocab)):
            inp_vectors[target_vocab[v]] = sess.run(embed, feed_dict={train_inputs: [wti[target_vocab[v]]]})
    
        # calculate similarities
        for v in inp_vectors:
            for vv in inp_vectors:
                sim_dict[v][vv].append(cosine_similarity(inp_vectors[v], inp_vectors[vv])[0][0])
        
        #collect time taken for run
        avg_time.append(time.time()-start_time)
        
        #close session
        sess.close()
     
    print('done training random')
    #after training, create output file with results    
    f = open(cwd+'/Results/business-science: {}_rand.txt'.format(target), 'w')
    
    
    f.write('\n\nSimilarities from rand Order: ')
    f.write('\n{}-{}: {} '.format(target,sense1,np.mean(sim_dict[target][sense1])))
    f.write('\n{}-{} STD: {}'.format(target,sense1, np.std(sim_dict[target][sense1])))
    f.write('\n\n{}-{}: {}'.format(target,sense2, np.mean(sim_dict[target][sense2])))
    f.write('\n{}-{} STD: {}'.format(target,sense2, np.std(sim_dict[target][sense2])))
    f.write('\n\n{} seconds'.format(np.mean(avg_time)))
    f.write('\n\n Number of Runs: {}'.format(num_runs))
    

    #close file
    f.close()
          
#Train NN in sense1-sense2 order
def train12(t1, t2, wti, itw, vocab_len, runs, embed_size, lr, negsamps,   target, sense1, sense2):
    

    print('training {}-{}...'.format(sense1,sense2))
    
    # number of runs
    num_runs = runs
    
    # embedding size
    embedding_size = embed_size
    
    #create smaller vocab out of target homonyms
    target_vocab = [target,sense1,sense2]
    
    avg_time = []
    
    # create a 2d-dictionary with list
    sim_dict = defaultdict(dict)
    
    # initialize the similarity dictionary
    for v in target_vocab:
        for ve in target_vocab:
            sim_dict[v][ve] = []
    

    #Topic1 then topic 2
    np.random.shuffle(t1)
    np.random.shuffle(t2)
    data1 = t1
    data2 = t2 
    
    #create feed for task 1
    input_feed_task1 = []
    output_feed_task1 = []
    

    for c in data1:
        splitted = c.split()
        input_feed_task1.append(wti[splitted[0]])
        output_feed_task1.append(wti[splitted[1]])
    
    # reshape the lists
    input_feed_task1 = np.array(input_feed_task1)
    output_feed_task1 = np.reshape(np.array(output_feed_task1), (len(output_feed_task1), 1))
    
    #find instances of target vocab in input_feed_task1 for faster fisher calculation
    target_vnum =[]
    
    for v in range(0, len(target_vocab)):
        target_vnum.append(wti[target_vocab[v]])
     
    target_ind = []
    for i in input_feed_task1:
        if i == target_vnum[0]:
            target_ind.append(i)
        if i == target_vnum[1]:
            target_ind.append(i)
        if i == target_vnum[2]:
            target_ind.append(i)
    
    #create feed for task 2
    input_feed_task2 = []
    output_feed_task2 = []
    for c in data2:
        splitted = c.split()
        input_feed_task2.append(wti[splitted[0]])
        output_feed_task2.append(wti[splitted[1]])
    
    # reshape the lists
    input_feed_task2 = np.array(input_feed_task2)
    output_feed_task2 = np.reshape(np.array(output_feed_task2), (len(output_feed_task2), 1))

    # for each run
    for i in range(0, num_runs):
        #start timer
        start_time = time.time()   
        
        #reset default tensorflow graph
        tf.reset_default_graph()

        
        # input and output placeholders
        train_inputs = tf.placeholder(tf.int32, shape = [1]) #shape = [batchsize]
        train_labels = tf.placeholder(tf.int32, shape = [1, 1])
        
        #initialize embedding matrix
        embeddings = tf.Variable(tf.random_uniform([vocab_len, embedding_size], -1.0, 1.0, seed = i))
        
        # collect embeddings    
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        # weight and bias matrix
        weights = tf.Variable(tf.truncated_normal([vocab_len, embedding_size], stddev = 1.0/np.sqrt(embedding_size), seed = i))
        biases = tf.Variable(tf.zeros([vocab_len]))

        #initialize loss function: nce
        loss = tf.reduce_mean(
        tf.nn.nce_loss(weights= weights,
                       biases= biases,
                       labels = train_labels,
                       inputs = embed,
                       num_sampled= negsamps,
                       num_classes= vocab_len)
        )
    
        #initialize optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        
        
        #initialize tensorflow session
        sess = tf.Session()
        
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # update weights for task 1
        for e in range(0, len(input_feed_task1)): #for all items in input
            x = sess.run([loss, optimizer], feed_dict = {train_inputs: [input_feed_task1[e]], train_labels: [output_feed_task1[e]]})


        # update weights for task 2
        for e in range(0, len(input_feed_task2)): #for all items in input
            x = sess.run([loss, optimizer], feed_dict = {train_inputs: [input_feed_task2[e]], train_labels: [output_feed_task2[e]]})
        
        # collect vectors 
        inp_vectors = {}
        for v in range(0, len(target_vocab)):
            inp_vectors[target_vocab[v]] = sess.run(embed, feed_dict={train_inputs: [wti[target_vocab[v]]]})
    
        # calculate similarities
        for v in inp_vectors:
            for vv in inp_vectors:
                sim_dict[v][vv].append(cosine_similarity(inp_vectors[v], inp_vectors[vv])[0][0])
                
        #collect time taken for each run
        avg_time.append(time.time()-start_time)
        
        #close session
        sess.close()
      
    print('done training {}-{}'.format(sense1,sense2))
    #create file to hold final results from model training    
    f = open(cwd+'/Results/business-health: {}_1-2.txt'.format(target), 'w')
    
    
    f.write('\n\nSimilarities from {}-{} Order: '.format(sense1, sense2))
    f.write('\n{}-{}: {} '.format(target,sense1,np.mean(sim_dict[target][sense1])))
    f.write('\n{}-{} STD: {}'.format(target,sense1, np.std(sim_dict[target][sense1])))
    f.write('\n\n{}-{}: {}'.format(target,sense2, np.mean(sim_dict[target][sense2])))
    f.write('\n{}-{} STD: {}'.format(target,sense2, np.std(sim_dict[target][sense2])))
    f.write('\n\n{} seconds'.format(np.mean(avg_time)))
    f.write('\n\n Number of Runs: {}'.format(num_runs))
    

    #close file
    f.close()
          

#Train NN in sense2-sense1 order
def train21(t1, t2, wti, itw, vocab_len, runs, embed_size, lr, negsamps,   target, sense1, sense2):
    # number of runs
    print('training {}-{}...'.format(sense2,sense1))
    num_runs = runs
    # embedding size
    embedding_size = embed_size
    
    target_vocab = [target,sense1,sense2]
    

    
        # create a 2d-dictionary with list
    sim_dict = defaultdict(dict)
    
    # initialize the similarity dictionary
    for v in target_vocab:
        for ve in target_vocab:
            sim_dict[v][ve] = []

    #Topic 2 then topic 1
    np.random.shuffle(t1)
    np.random.shuffle(t2)
    data1 = t2
    data2 = t1

    #create feed for task 1
    input_feed_task1 = []
    output_feed_task1 = []
    

    for c in data1:
        splitted = c.split()
        input_feed_task1.append(wti[splitted[0]])
        output_feed_task1.append(wti[splitted[1]])
    
    # reshape the lists
    input_feed_task1 = np.array(input_feed_task1)
    output_feed_task1 = np.reshape(np.array(output_feed_task1), (len(output_feed_task1), 1))
    
    #find instances of target vocab in input_feed_task1 for fisher calculation
    target_vnum =[]
    
    for v in range(0, len(target_vocab)):
        target_vnum.append(wti[target_vocab[v]])
     
    target_ind = []
    for i in input_feed_task1:
        if i == target_vnum[0]:
            target_ind.append(i)
        if i == target_vnum[1]:
            target_ind.append(i)
        if i == target_vnum[2]:
            target_ind.append(i)
    
    
    #create feed for task 2
    input_feed_task2 = []
    output_feed_task2 = []
    for c in data2:
        splitted = c.split()
        input_feed_task2.append(wti[splitted[0]])
        output_feed_task2.append(wti[splitted[1]])
    
    # reshape the lists
    input_feed_task2 = np.array(input_feed_task2)
    output_feed_task2 = np.reshape(np.array(output_feed_task2), (len(output_feed_task2), 1))
    
    
    avg_time = []
    # for each run
    for i in range(0, num_runs):
        #start timer
        start_time = time.time()

        #reset default tensorflow graph
        tf.reset_default_graph()

        # input and output placeholders
        train_inputs = tf.placeholder(tf.int32, shape = [1]) #shape = [batchsize]
        train_labels = tf.placeholder(tf.int32, shape = [1, 1])
        
        #initialize embedding matrix
        embeddings = tf.Variable(tf.random_uniform([vocab_len, embedding_size], -1.0, 1.0, seed = i))
        
        # collect embeddings    
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        # weight and bias matrix
        weights = tf.Variable(tf.truncated_normal([vocab_len, embedding_size], stddev = 1.0/np.sqrt(embedding_size), seed = i))
        biases = tf.Variable(tf.zeros([vocab_len]))
  
        #initialize loss function: nce
        loss = tf.reduce_mean(
        tf.nn.nce_loss(weights= weights,
                       biases= biases,
                       labels = train_labels,
                       inputs = embed,
                       num_sampled= 1,
                       num_classes= vocab_len)
        )
    
        #initialize optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        
        
        #initialize tensorflow session
        sess = tf.Session()
        
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # update weights for task 1
        for e in range(0, len(input_feed_task1)): #for all items in input
            x = sess.run([loss, optimizer], feed_dict = {train_inputs: [input_feed_task1[e]], train_labels: [output_feed_task1[e]]})


        
        # update weights for task 2
        for e in range(0, len(input_feed_task2)): #for all items in input
            x = sess.run([loss, optimizer], feed_dict = {train_inputs: [input_feed_task2[e]], train_labels: [output_feed_task2[e]]})
        
        # collect vectors 
        inp_vectors = {}
        for v in range(0, len(target_vocab)):
            inp_vectors[target_vocab[v]] = sess.run(embed, feed_dict={train_inputs: [wti[target_vocab[v]]]})
    
        # calculate similarities
        for v in inp_vectors:
            for vv in inp_vectors:
                sim_dict[v][vv].append(cosine_similarity(inp_vectors[v], inp_vectors[vv])[0][0])
                
        #collect time taken for each run
        avg_time.append(time.time()-start_time)
        
        #close session
        sess.close()        
    
    print('done training {}-{}'.format(sense2,sense1))
    f = open(cwd+'/Results/business-health: {}_2-1.txt'.format(target), 'w')
    
    
    f.write('\n\nSimilarities from {}-{} Order: '.format(sense2, sense1))
    f.write('\n{}-{}: {} '.format(target,sense1,np.mean(sim_dict[target][sense1])))
    f.write('\n{}-{} STD: {}'.format(target,sense1, np.std(sim_dict[target][sense1])))
    f.write('\n\n{}-{}: {}'.format(target,sense2, np.mean(sim_dict[target][sense2])))
    f.write('\n{}-{} STD: {}'.format(target,sense2, np.std(sim_dict[target][sense2])))
    f.write('\n\n{} seconds'.format(np.mean(avg_time)))
    f.write('\n\n Number of Runs: {}'.format(num_runs))

    f.close()
          
    sess.close()    
    

def run(f1,f2, runs, embed_size, lr, negsamps, target, sense1, sense2):
    '''f1 = filepath 1, f2 = filepath2, runs = number of runs per core, embed_size = size of network,
     lr = learning rate, negsamps = number of negative samples for nce loss to use,
     target = homonym, sense1 = first sense pure synonym, sense 2 = second sense pure synonym'''
    
    file1, file2 = read_data(f1, f2)  
    # collect topic 1 sentences
    topic1_sent = seperate_sentences(fetch_clean_data(file1)[0])
    # collect topic 2 sentences
    topic2_sent = seperate_sentences(fetch_clean_data(file2)[0])
       
    
    vocab = list(get_vocab(topic1_sent)) + list(get_vocab(topic2_sent))
    word_to_index, index_to_word = {}, {}
    for v in range(0, len(vocab)):
        word_to_index[vocab[v]] = v
        index_to_word[v] = vocab[v]
        
    
    #Use multithreading
    #train nn on random order
    p1 = Process(target=trainrand, args=(topic1_sent, topic2_sent, word_to_index, index_to_word,len(vocab),
                                         runs, embed_size, lr,negsamps, target,sense1,sense2))
    p1.start()
    
    #train nn on sense1-sense2 order
    p2 = Process(target= train12, args=(topic1_sent, topic2_sent, word_to_index, index_to_word,len(vocab), 
                                        runs, embed_size, lr,negsamps,  target,sense1,sense2))
    p2.start()
    
    #train nn on sense2-sense1 order
    p3 = Process(target=train21, args=(topic1_sent, topic2_sent, word_to_index, index_to_word,len(vocab),
                                       runs, embed_size, lr, negsamps,  target,sense1,sense2))
    p3.start()
    
    #join threads
    p1.join()
    p2.join()
    p3.join()




if __name__ == '__main__':

    import tensorflow as tf
    import numpy as np
    from collections import defaultdict
    from sklearn.metrics.pairwise import cosine_similarity
    import pickle
    import string
    import collections
    from nltk.corpus import stopwords
    from multiprocessing import Process
    
    
    import time
    import os
    
    #filter out annoying cpu messages, yes I know I should build from source...
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    stopwords = set(stopwords.words("english"))
    
    cwd = os.getcwd()
    file1p = cwd + '/Source Texts/business.txt' 
    file2p = cwd + '/Source Texts/health.txt'
    
    #target homonym and senses depend on corpus files
    run(f1 =file1p, f2 = file2p, runs = 100, embed_size = 50,lr= 0.3, negsamps= 64,  
         target='firm', sense1='hire', sense2='connective')
    
