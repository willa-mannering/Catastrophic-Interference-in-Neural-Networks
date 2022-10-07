#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 14:54:37 2018

@author: willamannering
"""

import RVA as r #import RVA class
import measures as m
import numpy as np
import os
import time


#get current working directory
cwd = os.getcwd()


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


#add period to end of artificial sentences for RVA to read in via text file
def add_stop(sentences):
    new = []
    for sen in sentences:
        new.append(sen +'.')
    return new

#write pairs of words to text file
def create_corp(pathname, sentences):
    f = open(pathname, 'w')
    for sen in sentences:
        for word in sen:
            f.write(word)
    f.close()



def train_model(dist, dimensions, num_runs, corpus_path, word1 = 'bass' , word2 = 'trout', word3 = 'acoustic'):
    print('Training model with data from {}'.format(corpus_path))
    
    runtime = [] #initialize array to save run time info
    rvaSims12 = [] #initialize array to save similarity calculations for target-sense1
    rvaSims13 = [] #initialize array to save similarity calculations for target-sense2

    for i in range(num_runs):
        start_time = time.time()
        rva = r.RVA(distribution = dist,dims = dimensions) #initialize RVA object

        rva.read(corpus_path, accumulateFunction=r.accumulate.randomVector,  context = 'sen') #accumulate vectors

        rvaSims12.append(m.sim.cosineMatrix(rva, [word1, word2]))#find similarity for target-sense1
        rvaSims13.append(m.sim.cosineMatrix(rva, [word1, word3]))#find similarity for target-sense2
        print(i)
        runtime.append(time.time()-start_time)#collect runtime
        
    #collect avg sims
    sims13 = []
    sims12 = []
    for i in range(num_runs):
        sims13.append(rvaSims13[i][0][word3][word1])  
        sims12.append(rvaSims12[i][0][word1][word2])  
        
    #return avg sims, standard deviation, runtime           
    return [np.mean(sims12), np.std(sims12), np.mean(sims13), np.std(sims13)], runtime

#%%
    
#-------------------Create Artificial Corpus text files-----------------#
    
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
    
# if dominant = True, the data is considered a dominant sense
# if dominant = False, the data is considered a subordinate sense
fish = generate_dom_sub(fish_data, dominant = True, reps = n_sent_reps)
instrument = generate_dom_sub(instrument_data, dominant = False, reps = n_sent_reps)

# shuffle each sense
np.random.shuffle(fish)
np.random.shuffle(instrument)

# create random case
rand = instrument + fish
np.random.shuffle(rand)
random = add_stop(rand)
    
# create instrument then fish case
infish = instrument + fish
inst_fish = add_stop(infish)
    
# create fish then instrument case
fishin = fish + instrument
fish_inst = add_stop(fishin)

#create_corp(cwd+'/rand.txt', random)
#create_corp(cwd+'/if.txt', infish)
#create_corp(cwd+'/fi.txt', fishin)
#
create_corp(cwd+'/rand_fishdom.txt', random)
create_corp(cwd+'/if_fishdom.txt', inst_fish)
create_corp(cwd+'/fi_fishdom.txt', fish_inst)
##
#create_corp(cwd+'/rand_indom.txt', random)
#create_corp(cwd+'/if_indom.txt', infish)
#create_corp(cwd+'/fi_indom.txt', fishin)
    
#%%

#------------------- Train Models -----------------------------
runs = 10
dims = 35

#Run for no dom
#random_sims, rand_run = train_model('gaussian', dims, runs, cwd+'/rand.txt')
#fish_in_sims, fi_run = train_model('gaussian', dims, runs, cwd+'/fi.txt')
#in_fish_sims, if_run = train_model('gaussian', dims, runs, cwd+'/if.txt')

#Rund for fishdom
random_sims, rand_run = train_model('gaussian', dims, runs, cwd+'/rand_fishdom.txt')
fish_in_sims, fi_run = train_model('gaussian', dims, runs, cwd+'/fi_fishdom.txt')
in_fish_sims, if_run = train_model('gaussian', dims, runs, cwd+'/if_fishdom.txt')

#Run for instrument dom
#random_sims, rand_run = train_model('gaussian', dims, runs, cwd+'/rand_indom.txt')
#fish_in_sims, fi_run = train_model('gaussian', dims, runs, cwd+'/fi_indom.txt')
#in_fish_sims, if_run = train_model('gaussian', dims, runs, cwd+'/if_indom.txt')


#Output for random order
print('\nSimilarities from Random Order: ')
print('Bass-Trout : {} \n Bass-Acoustic {}'.format(random_sims[0], random_sims[2]))
print('Bass-Trout STD: {} \n Bass-Acoustic STD{}'.format(random_sims[1], random_sims[3]))
print('\nAverage Runtime {} seconds'.format(np.mean(rand_run)))


#Output for fish-instrument order
print('\nSimilarities from Fish-Instrument Order: ')
print('Bass-Trout : {} \n Bass-Acoustic {}'.format(fish_in_sims[0], fish_in_sims[2]))
print('Bass-Trout STD: {} \n Bass-Acoustic STD{}'.format(fish_in_sims[1], fish_in_sims[3]))
print('\nAverage Runtime {} seconds'.format(np.mean(fi_run)))

#Output for instrument-fish order
print('\nSimilarities from Instrument-Fish Order: ')
print('Bass-Trout : {} \n Bass-Acoustic {}'.format(in_fish_sims[0], in_fish_sims[2]))
print('Bass-Trout STD: {} \n Bass-Acoustic STD{}'.format(in_fish_sims[1], in_fish_sims[3]))
print('\nAverage Runtime {} seconds'.format(np.mean(if_run)))


