'''
Experiment 2a - CI in artificial language using EWC and neural net

author: @willamannering
date: 10/29/2019

'''

#################### Set up Vocab + EWC functions ##############################3

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


def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]
   

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


def compute_fisher(train_inputs, train_labels, input_feed, sess, num_iter, var_list, output_layer): #variable list, y = output(y_train), x=x tensor
    '''train_inputs = input words to model
        train_labels = correct output for input
        input_feed = 
        sess = tensorflow session
        num_iter = number of samples to take of vocab
        var_list = weights and biases
        output_layer = softmax layer of model'''

    F_accum = []
    for v in range(len(var_list)):
        F_accum.append(np.zeros(var_list[v].get_shape().as_list())) 
        
    probs = tf.nn.softmax(output_layer) #computes probability of each class   
    class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
     
    for i in range(num_iter): # aka do this 200 times, could increase...
        # select random word from input_feed
        word_ind = np.random.randint(len(input_feed))
        
        # compute first-order derivatives
        # uses words in input_feed
        #ders, computes first derivative of log probability of chosen class for each variable
        ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), var_list), feed_dict={train_inputs: input_feed[word_ind:word_ind+1]})
        # square each derivative and add to total
        for v in range(len(F_accum)):
            F_accum[v] += np.square(ders[v])

    # divide totals by number of samples; save in class
    for v in range(len(F_accum)):
        F_accum[v] /= num_iter
        
    return F_accum,probs


def set_ewcloss(var_list, star_vars, lam, F_accum, old_loss):
    
    ewc_loss = old_loss
    
    for v in range(len(var_list)):
        ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(F_accum[v].astype(np.float32),tf.square(var_list[v] - star_vars[v])))
        
    return ewc_loss




################### Train EWC Model########################################
# number of runs
def train_ewc(word_to_index, index_to_word, runs, embed_size, lr, vocab, cond, fish, instrument):
    num_runs = runs
    
    avg_time = []
    # embedding size
    embedding_size = embed_size
    
    # create a 2d-dictionary with list
    sim_dict = defaultdict(dict)
    
    # initialize the similarity dictionary
    for v in vocab:
        for ve in vocab:
            sim_dict[v][ve] = []
            
    # for each run
    for i in range(0, num_runs):
        start_time = time.time()

        #Fish then acoustic ordering    
        if cond == 1:
            #fish-acoustic
            np.random.shuffle(fish)
            np.random.shuffle(instrument)
            data1 = fish
            data2 = instrument
            
        #Acoustic then fish ordering
        if cond == 2:
            np.random.shuffle(fish)
            np.random.shuffle(instrument)
            data1 = instrument
            data2 = fish
            
        #random ordering
        elif cond == 3:
            
            d = fish+instrument
            np.random.shuffle(d)
            data1 = split_list(d)[0]
            data2 = split_list(d)[1]
            
            
        # reset graph at every run
        print (i)
        tf.reset_default_graph()
        
        # embedding matrix
        
            # input and output placeholders
        train_inputs = tf.placeholder(tf.int32, shape = [1]) #shape = [batchsize]
        train_labels = tf.placeholder(tf.int32, shape = [1, 1])
    
        embeddings = tf.Variable(tf.random_uniform([len(vocab), embedding_size], -1.0, 1.0, seed = i))
           # collect embeddings    
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        # weight matrix
        weights = tf.Variable(tf.truncated_normal([len(vocab), embedding_size], stddev = 1.0/np.sqrt(embedding_size), seed = i))
        biases = tf.Variable(tf.zeros([len(vocab)]))
        hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases
        varlist = [weights, biases]
    
        train_one_hot = tf.one_hot(train_labels, len(vocab))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))
 
    
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        
        
        #-----------Task 1 -------------------
        # create input and output feeds for just task 1
        input_feed_task1 = []
        output_feed_task1 = []
        for c in data1:
            splitted = c.split()
            input_feed_task1.append(word_to_index[splitted[0]])
            output_feed_task1.append(word_to_index[splitted[1]])
        
        # reshape the lists
        input_feed_task1 = np.array(input_feed_task1)
        output_feed_task1 = np.reshape(np.array(output_feed_task1), (len(output_feed_task1), 1))
        
        # initialize tensorflow session
        sess = tf.Session()
        
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # update weights after example
        for e in range(0, len(input_feed_task1)): #for all items in input
            x = sess.run(optimizer, feed_dict = {train_inputs: [input_feed_task1[e]], train_labels: [output_feed_task1[e]]})
    
        #compute fisher info
        fisher = compute_fisher(train_inputs, train_labels, input_feed_task1, sess, 50, varlist , hidden_out)
        
        
        #save weights
        weights_saved = weights
        biases_saved = biases
        
        saved_vars = [weights_saved, biases_saved]
        for v in range(0,len(varlist)):
            sess.run(varlist[v].assign(saved_vars[v]))
   
        #------------Task 2------------------

        #update loss function
        loss2 = set_ewcloss(varlist,saved_vars, 15, fisher, loss)
        
        #update optimizer to use new loss function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss2)
        
        #create input and output feeds for task 2
        input_feed_task2 = []
        output_feed_task2 = []
        for c in data2:
            splitted = c.split()
            input_feed_task2.append(word_to_index[splitted[0]])
            output_feed_task2.append(word_to_index[splitted[1]])
        
        # reshape the lists
        input_feed_task2 = np.array(input_feed_task2)
        output_feed_task2 = np.reshape(np.array(output_feed_task2), (len(output_feed_task2), 1))
    
        
        # update weights after example
        for e in range(0, len(input_feed_task2)): #for all items in input
            x = sess.run(optimizer, feed_dict = {train_inputs: [input_feed_task2[e]], train_labels: [output_feed_task2[e]]})
        
        # collect vectors 
        inp_vectors = {}
        for v in range(0, len(vocab)):
            inp_vectors[vocab[v]] = sess.run(embed, feed_dict={train_inputs: [word_to_index[vocab[v]]]})
    
        # calculate similarities
        for v in inp_vectors:
            for vv in inp_vectors:
                sim_dict[v][vv].append(cosine_similarity(inp_vectors[v], inp_vectors[vv])[0][0])
                
    
        avg_time.append(time.time()-start_time)
        
    print('EWC Condition {}'.format(cond))
    print ('Bass - Acoustic cond {}: '.format(cond), np.mean(sim_dict['bass']['acoustic']))
    print ('Bass - Trout cond {}: '.format(cond), np.mean(sim_dict['bass']['trout']))
    
    print ('Bass - Acoustic Std cond {}: '.format(cond), np.std(sim_dict['bass']['acoustic']))
    print ('Bass - Trout Std cond {}: '.format(cond), np.std(sim_dict['bass']['trout']))
    
    print(np.mean(avg_time), ' seconds')

    sess.close()
    
    

################### Train Basic Neural Net########################################
# number of runs
def train_nn(word_to_index, index_to_word, runs, embed_size, lr, vocab, cond, fish, instrument):
    num_runs = runs
    
    avg_time = []
    # embedding size
    embedding_size = embed_size
    
    # create a 2d-dictionary with list
    sim_dict = defaultdict(dict)
    
    # initialize the similarity dictionary
    for v in vocab:
        for ve in vocab:
            sim_dict[v][ve] = []
    # for each run
    for i in range(0, num_runs):
        start_time = time.time()    

        #Fish then acoustic ordering
        if cond == 1:
            
            np.random.shuffle(fish)
            np.random.shuffle(instrument)
            data1 = fish
            data2 = instrument
            
        #Acoustic then fish ordering
        elif cond == 2:
            
            np.random.shuffle(fish)
            np.random.shuffle(instrument)
            data1 = instrument
            data2 = fish
            
        #Random ordering condition   
        if cond == 3:
            
            d = fish+instrument
            np.random.shuffle(d)
            data1 = split_list(d)[0]
            data2 = split_list(d)[1]
            
        # reset graph at every run
        print (i)
        tf.reset_default_graph()
        
        # embedding matrix
        # input and output placeholders
        train_inputs = tf.placeholder(tf.int32, shape = [1]) #shape = [batchsize]
        train_labels = tf.placeholder(tf.int32, shape = [1, 1])
    
        embeddings = tf.Variable(tf.random_uniform([len(vocab), embedding_size], -1.0, 1.0, seed = i))
        
        # collect embeddings    
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        # weight matrix
        weights = tf.Variable(tf.truncated_normal([len(vocab), embedding_size], stddev = 1.0/np.sqrt(embedding_size), seed = i))
        biases = tf.Variable(tf.zeros([len(vocab)]))
        hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases
        
    
        train_one_hot = tf.one_hot(train_labels, len(vocab))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))
    
    
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        
        
        #-----------Task 1 -------------------
        # create input and output feeds for task 1
        input_feed_task1 = []
        output_feed_task1 = []
        for c in data1:
            splitted = c.split()
            input_feed_task1.append(word_to_index[splitted[0]])
            output_feed_task1.append(word_to_index[splitted[1]])
        
        # reshape the lists
        input_feed_task1 = np.array(input_feed_task1)
        output_feed_task1 = np.reshape(np.array(output_feed_task1), (len(output_feed_task1), 1))
        
        # initialize tensorflow session
        sess = tf.Session()
        
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # update weights after example
        for e in range(0, len(input_feed_task1)): #for all items in input
            x = sess.run(optimizer, feed_dict = {train_inputs: [input_feed_task1[e]], train_labels: [output_feed_task1[e]]})
    
    
     
        #------------Task 2------------------
        #create input and output feeds for task 2
        input_feed_task2 = []
        output_feed_task2 = []
        for c in data2:
            splitted = c.split()
            input_feed_task2.append(word_to_index[splitted[0]])
            output_feed_task2.append(word_to_index[splitted[1]])
        
        # reshape the lists
        input_feed_task2 = np.array(input_feed_task2)
        output_feed_task2 = np.reshape(np.array(output_feed_task2), (len(output_feed_task2), 1))
    
        
        # update weights after example
        for e in range(0, len(input_feed_task2)): #for all items in input
            x = sess.run(optimizer, feed_dict = {train_inputs: [input_feed_task2[e]], train_labels: [output_feed_task2[e]]})
        
        # collect vectors 
        inp_vectors = {}
        for v in range(0, len(vocab)):
            inp_vectors[vocab[v]] = sess.run(embed, feed_dict={train_inputs: [word_to_index[vocab[v]]]})
    
        # calculate similarities
        for v in inp_vectors:
            for vv in inp_vectors:
                sim_dict[v][vv].append(cosine_similarity(inp_vectors[v], inp_vectors[vv])[0][0])
                
    
        avg_time.append(time.time()-start_time)
    print('Vanilla Condition {}'.format(cond))
    print ('Bass - Acoustic: {}: '.format(cond), np.mean(sim_dict['bass']['acoustic']))
    print ('Bass - Trout: ', np.mean(sim_dict['bass']['trout']))
    
    print ('Bass - Acoustic Std: ', np.std(sim_dict['bass']['acoustic']))
    print ('Bass - Trout Std: ', np.std(sim_dict['bass']['trout']))
    
    print(np.mean(avg_time), ' seconds')

    
    sess.close()
    
    

    
    
def run(runs, embed_size, lr):

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
        
        
    #use multithreading
    #run ewc cond = 1 fish then acoustic order
    p1 = Process(target= train_ewc, args=(word_to_index, index_to_word, runs, embed_size, lr, vocab, 1, fish, instrument))
    p1.start()
    
    #run ewc cond = 2 acoustic then fish order
    p2 = Process(target= train_ewc, args=(word_to_index, index_to_word,
                                      runs, embed_size, lr, vocab, 2, fish, instrument))
    p2.start()
    
    #run nn cond 1 = fish then acoustic order
    p3 = Process(target= train_nn, args=(word_to_index, index_to_word,
                                        runs, embed_size, lr, vocab, 1, fish, instrument))
    p3.start()
    
    #run nn cond 2 = acoustic then fish order
    p4 = Process(target= train_nn, args=(word_to_index, index_to_word,
                                        runs, embed_size, lr, vocab, 2, fish, instrument))
    p4.start()
    
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    

if __name__ == '__main__':

    import tensorflow as tf
    import numpy as np
    from collections import defaultdict
    from sklearn.metrics.pairwise import cosine_similarity    
    import time
    from multiprocessing import Process
    
    
    run(runs = 100, embed_size = 10,lr= 0.02)
  
    
    
    
