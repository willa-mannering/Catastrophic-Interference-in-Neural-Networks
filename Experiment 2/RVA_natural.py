#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:27:13 2018

Train RVA on natural TASA corpus
       
@author: willamannering
"""

import RVA as r #import RVA class
import measures as m
import numpy as np
import time
import os


#indicate file location of corpus to use
cwd = os.getcwd()


def train_model(dist, dimensions, num_runs, corpus_path, saved_vecs = False, word1 = 'bass' , word2 = 'trout', word3 = 'acoustic'):
    ''' dist = distribution, dimensions = dimensions, num_runs = number of runs, corpus_path = path to saved corpus,
        saved_vecs = whether vectors are saved already or not, word1 = target, word2 = sense1, word3 = sense3 '''
        
    print('Training model with data from {}'.format(corpus_path))
    
    runtime = [] #initialize array to save run time info
    rvaSims12 = [] #initialize array to save similarity calculations for target-sense1
    rvaSims13 = [] #initialize array to save similarity calculations for target-sense2

    for i in range(num_runs):
        start_time = time.time() #start timing function
        rva = r.RVA(distribution = dist,dims = dimensions) #create RVA object
        
        #if vectors previously saved
        if saved_vecs:
            rva = rva.load(corpus_path+'_vectors') #load vectors
            rvaSims12.append(m.sim.cosineMatrix(rva, [word1, word2])) #find similarity for target-sense1
            rvaSims13.append(m.sim.cosineMatrix(rva, [word1, word3])) #find similarity for target-sense2
            runtime.append(time.time()-start_time) #collect runtime
            
        #if vectors not saved
        else:
            rva.read_saved(corpus_path, accumulateFunction=r.accumulate.randomVector, context = 'sen') #call read function to accumulate vectors
        
            rva.save(corpus_path + '_vectors') #save vectors

            rvaSims12.append(m.sim.cosineMatrix(rva, [word1, word2]))#find similarity for target-sense1
            rvaSims13.append(m.sim.cosineMatrix(rva, [word1, word3]))#find similarity for target-sense2
            print(i)
            runtime.append(time.time()-start_time) #collect runtime

    #find average similarities
    sims13 = []
    sims12 = []
    for i in range(num_runs):
        sims13.append(rvaSims13[i][0][word3][word1])  
        sims12.append(rvaSims12[i][0][word1][word2])  
        
    #return avg sims, standard deviation, runtime   
    return [np.mean(sims12), np.std(sims12), np.mean(sims13), np.std(sims13)], runtime


def run(dimensions, num_runs, words, topics, saved=False):
    ''' words = [target,sense1,sense2], topics = [tasa corp1, tasa corp2]'''
    
    #set runs and dims
    runs = num_runs
    dims = dimensions
    
    #set target, sense1,sense2
    target = words[0]
    sense1 = words[1]
    sense2 = words[2]
    
    top1 = topics[0]
    top2 = topics[1]
    
    #train rva model on randomly ordered corpus
    random_sims, rand_run = train_model('gaussian', dims, runs, cwd+'/preprocessed_texts/Vectors/{}-{}_RandOrder.pkl'.format(top1,top2), saved, target, sense1, sense2)
    
    #train rva on sense1-sense2 ordered corpus
    top1_top2_sims, onetwo_run = train_model('gaussian', dims, runs, cwd + '/preprocessed_texts/Vectors/{}-{}_1-2.pkl'.format(top1,top2), saved, target, sense1, sense2)
    
    #train rva on sense2-sense1 ordered corpus
    top2_top1_sims, twoone_run = train_model('gaussian', dims, runs, cwd + '/preprocessed_texts/Vectors/{}-{}_2-1.pkl'.format(top1,top2), saved, target, sense1, sense2)

    #create ouptut file for similarity results
    f = open(cwd+'/Results/{}-{}: similarities for senses of {}.txt'.format(top1,top2, target), 'w')
    
    
    #write to file
    f.write('\nSimilarities from Random Order: ')
    f.write('\n{}-{} : {} \n{}-{} : {}'.format(target, sense1, random_sims[0], target, sense2, random_sims[2]))
    f.write('\nSTD {}-{} : {} \nSTD {}-{} : {}'.format(target, sense1, random_sims[1], target, sense2, random_sims[3]))
    f.write('\nAverage Runtime {} seconds'.format(np.mean(rand_run)))
    
    f.write('\n\nSimilarities from {}-{} Order: '.format(sense1, sense2))
    f.write('\n{}-{} : {} \n{}-{} : {}'.format(target, sense1, top1_top2_sims[0], target, sense2, top1_top2_sims[2]))
    f.write('\nSTD {}-{} : {} \nSTD {}-{} : {}'.format(target, sense1, top1_top2_sims[1], target, sense2, top1_top2_sims[3]))
    f.write('\nAverage Runtime {} seconds'.format(np.mean(onetwo_run)))
    
    f.write('\n\nSimilarities from {}-{} Order: '.format(sense2, sense1))
    f.write('\n{}-{} : {} \n{}-{} : {}'.format(target, sense1, top2_top1_sims[0], target, sense2, top2_top1_sims[2]))
    f.write('\nSTD {}-{} : {} \nSTD {}-{} : {}'.format(target, sense1, top2_top1_sims[1], target, sense2, top2_top1_sims[3]))
    f.write('\nAverage Runtime {} seconds'.format(np.mean(twoone_run)))
    
    #close file
    f.close()


#%%

#run 1 homonym
run(350, 10, ['hamper','hustling','unprofitable'], ['languagearts','socialstudies'],saved = True)

#all homonyms from TASA
torun = [['firm', 'hire', 'connective','business', 'health'],
         ['compact','parietal', 'separatists','science', 'socialstudies'],
         ['hull','breakwater', 'indignant','science', 'socialstudies'], 
         ['compound', 'reactive', 'predicates', 'science', 'languagearts'],
         ['pitch','strings','pitchers','science','languagearts'],
         ['cap','nasal','undershirt','science','languagearts'],
         ['gum', 'root','chewed', 'science','languagearts'], 
         ['bull','weasel','daybreak','languagearts','socialstudies'],
         ['pupil', 'iris', 'publishers', 'science','socialstudies'], 
         ['hamper','hustling','unprofitable', 'languagearts','socialstudies'],
         ['capital', 'honolulu', 'harry', 'languagearts', 'business'], 
         ['net', 'puck','expenses','languagearts','business'],
         ['slip','buckles','cashiers', 'science','business'], 
         ['plane', 'block','flies','industrialarts','socialstudies']]

#train all homonyms with saved vectors
for x in torun:
    run(350, 10, [x[0],x[1],x[2]], [x[3],x[4]], saved = True)














