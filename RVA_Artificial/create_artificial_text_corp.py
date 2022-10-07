
import numpy as np


def generate_dom_sub(data1, data2, dominant, num_sent):
    """generate artificial text data with equal senses or a dominant/subordinate sense

    Args:
        data (list): list of strings representing words in artificial text
        dominant (boolean): True = make this sense dominant, False = don't make this sense dominant
        num_sent (int): number of sentences to create

    Returns:
        _type_: _description_
    """    
    inner_list = []
    if dominant == True: #make num_sent number of sentences
        for d in data:
            inner_list += [d] * num_sent
    else:
        for d in data: #divide num_sent by 3
            inner_list += [d] * (int(num_sent/3))
    return inner_list

#add period to end of artificial sentences for RVA to read in via text file
def add_stop(sentences):
    new = []
    for sen in sentences:
        new.append(sen +'.')
    return new

#write pairs of words to text file
def create_corp(pathname, sentences):
    """creates files from sentence data generated from create_sents

    Args:
        pathname (string): path of file to save
        sentences (list): list of sentences to send to file
    """    
    f = open(pathname, 'w')
    for sen in sentences:
        for word in sen:
            f.write(word)
    f.close()

def create_sents(data1, data2, num_sent, dominant=None):
    """creates sentences for artificial corpus

    Args:
        data1 (list): list of strings for dataset 1
        data2 (list): list of strings for dataset 2
        dominant (string): choose which dataset is dominant
        num_sent (int): number of sentences to make
    """    
    d1, d2 = generate_dom_sub(data1, data2, dominant = True, reps = num_sent)
    d2 = generate_dom_sub(data2, dominant = False, reps = num_sent)

    # shuffle each sense
    np.random.shuffle(d1)
    np.random.shuffle(d2)

    # create random case
    rand = d1 + d2
    np.random.shuffle(rand)
    random = add_stop(rand)

    # create data1 then data2 case
    d1d2 = d1 + d2
    d1_d2 = add_stop(d1d2)

    # create fish then instrument case
    d2d1 = d2 + d1
    d2_d1 = add_stop(d2d1)

    return random, d1_d2, d2_d1