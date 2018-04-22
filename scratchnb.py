#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import codecs
import sys
import os.path
from math import log, sqrt

#for random probs and train-test split
import numpy as np
#loading data
import pandas as pd
#nltk for preprocessing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#saving dicts
import pickle
#visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#python -m nltk.downloader 'punkt'

train_path = "./train/train.tsv"
testPath = "./dev-0/"
trainingBatchSize = 886699
traintotestratio = 1


sad_words_dict = dict()
happy_words_dict = dict()
Ps = 0.0
Ph = 0.0
happy_count = 0
sad_count = 0

#gets sentences from file - if line is corrupted returns empty string
def get_texts(filename, with_label=True,maxtrainsize=10000):
    print ">Loading sentences"
    all_texts = []
    with codecs.open(filename, encoding = "utf-8" ) as fp:
        for line in fp:
            if with_label:
                splitted = line.split("\t")
                if len(splitted) == 1:
                    line = str('').encode("utf-8") 
                else:
                    line = line.split("\t")[1]
            all_texts.append(line)
            if len(all_texts) >= maxtrainsize:
                return all_texts
    return all_texts


def get_labels(filename, with_text=True, maxtrainsize=10000):
    print ">Loading labels"
    all_labels = []
    with codecs.open(filename,  encoding = "utf-8" ) as fp:
        for line in fp:
            if with_text:
                line = line.split("\t")[0]
            all_labels.append(getFloat(line))
            if len(all_labels) >= maxtrainsize:
                return all_labels 
    return all_labels

def simple_preprocess(sentence):
    sentence =  sentence.lower().replace("emoticon","").replace("<","").replace(">","").replace(",","").replace(".","").replace("-"," ").replace("!","").replace("?","").replace("_"," ").replace(":"," ").replace(";"," ").replace(")","").replace("("," ").replace("\"","")
    words = word_tokenize(sentence)
    sw = stopwords.words('polish')
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join([word for word in words if word not in sw])

def creade_word_features(sentence,happy = True):
    words = sentence.split()
    if happy:
        if "sentences_counter" in happy_words_dict:
                happy_words_dict['sentences_counter'] += 1
        else:
            happy_words_dict["sentences_counter"] = 1
        for word in words:
            if word in happy_words_dict:
                happy_words_dict[word] += 1
            else:
                happy_words_dict[word] = 1     
    else:
        if "sentences_counter" in sad_words_dict:
                sad_words_dict['sentences_counter'] += 1
        else:
            sad_words_dict["sentences_counter"] = 1
        for word in words:
            if word in sad_words_dict:
                sad_words_dict[word] += 1
            else:
                sad_words_dict[word] = 1
    
def calculate_class_prob(testSentence):
    
    testSentence = simple_preprocess(testSentence.decode('utf-8'))

    Pth = []
    Pts = []
    words = testSentence.split()
    #happy
    for word in words:
        Tc = 1
        if word in happy_words_dict:
            Pth.append((happy_words_dict[word]+Tc)/float(happy_count * Ph + happy_count))
        else:
            Pth.append(1/float(happy_count * Ph + happy_count))
    #sad
    for word in words:
        Tc = 1
        if word in sad_words_dict:
            Pts.append((sad_words_dict[word]+Tc)/float(sad_count * Ps + sad_count))
        else:
            Pts.append(1/float(sad_count * Ps + sad_count))

    Pdh = 1.0 
    Pds = 1.0
    for i in range(len(Pth)):
        Pdh = Pth[i] * Pdh
        Pds = Pts[i] * Pds
    PdhPh = Pdh * Ph
    PdsPs = Pds * Ps
    return [PdhPh, PdsPs]

def train(maxtrainsize = 10000):
    #Load training data
    print ">Loading train data"
    X_train = get_texts(train_path,True,maxtrainsize)
    y_train = get_labels(train_path,True,maxtrainsize)

    #limit data
    print ">>Processing train data"
    #X_train = X_train[:10]
    #y_train = y_train[:10]


    print ">>Processing sentences"
    rng = len(X_train)
    for i in range(rng):
        if trainingBatchSize > 10000 and i%(trainingBatchSize/1000)==0:
            sys.stdout.write('\r' + str(i)+"/"+str(rng) + " sentences - " + str(i*100/float(rng)) + "%")
            sys.stdout.flush()            
        X_train[i] = simple_preprocess(X_train[i])
    sys.stdout.write('\r' + str(rng)+"/"+str(rng) + " sentences - 100%")
    sys.stdout.flush()   
    print ""

    dataset = pd.DataFrame({'sentence' : X_train,'label' : y_train})

    '''
    #Split into train and test
    totalSentences = dataset['sentence'].shape[0]
    trainIndex, testIndex = list(), list()
    #get randomized entries 
    for i in range(totalSentences):
        if np.random.uniform(0,1) <= traintotestratio:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainset= dataset.loc[trainIndex]
    testset = dataset.loc[testIndex]
    #reset indexes
    trainset.reset_index(inplace=True)
    trainset.drop(['index'],axis = 1, inplace = True)
    '''
    trainset = dataset
    #get word features
    print ">>Getting word features"

    for index, row in trainset.iterrows():
        if row['label'] > 0.5:
            creade_word_features(row['sentence'],True)
        else:
            creade_word_features(row['sentence'],False)
    
    save_dictionaries(happy_words_dict,sad_words_dict)

    return happy_words_dict, sad_words_dict

def set_apriori_probs():    
    #h - happy
    #s - sad

    print ">Getting apriori probs and word counts"

    sad_count = sad_words_dict['sentences_counter']
    happy_count = happy_words_dict['sentences_counter']
    Ph = happy_count/float(sad_count+happy_count)
    Ps = 1 - Ph

    print "Happy sentences: " + str(happy_count) + ", " + str(Ph) + "%"
    print "Sad sentences: " + str(sad_count) + ", " + str(Ps) + "%"
    return Ps,Ph, sad_count, happy_count

    
def save_dictionaries(happy_dict,sad_dict):
    with open('./train/happy_dict.pkl', 'wb') as h:
        pickle.dump(happy_words_dict, h, pickle.HIGHEST_PROTOCOL)
    with open('./train/sad_dict.pkl', 'wb') as s:
        pickle.dump(sad_words_dict, s, pickle.HIGHEST_PROTOCOL)
    

def load_dictionaries():
    with open('./train/happy_dict.pkl', 'rb') as h:
        happy_words_dict = pickle.load(h)
    with open('./train/sad_dict.pkl', 'rb') as s:
        sad_words_dict = pickle.load(s)
    return happy_words_dict,sad_words_dict

def getFloat(s):
    try:
        return float(s)
    except ValueError:
        return 0.5

if os.path.isfile('./train/happy_dict.pkl'):
    print "===Loading saved dictionaries"
    happy_words_dict, sad_words_dict = load_dictionaries()
else:
    print "===Training new model"
    happy_words_dict, sad_words_dict = train(maxtrainsize=trainingBatchSize)        

Ps, Ph, sad_count, happy_count = set_apriori_probs()
       


print "===Predicting"
insentences = get_texts(testPath + "in.tsv",with_label=False,maxtrainsize=99999999)

print ">Calculating probabilites"
results = []
for line in insentences:
    results.append(calculate_class_prob(line.encode('utf-8')))


print "===Saving results"
with open(testPath + "/out.tsv", 'w') as outtsv:
  prob = 0.5
  for y in results:
      if y[0] < y[1]:
        prob= 0.3
      else:
        prob=0.7
      outtsv.write(str(prob) + '\n')
print "===Done."
