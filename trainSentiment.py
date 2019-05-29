import os
import re
import sys
import numpy as np
import pandas as pd

import collections
import nltk.classify.util, nltk.metrics
from  nltk.metrics.scores import precision, recall, f_measure
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
import random
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
# using cPickale for serialization of objects
import pickle
from commonUtils import getSentimentDictionary, getAllFeaturesList
from sentimentUtils import getFeatureVector, getSentiment,extract_word_features



if not os.path.exists('pkl'):
        os.makedirs('pkl') 

#put Name of all Serialized objects
SentimentDictPkl = os.path.join('pkl','SentimentDict.pkl')
StopWordsPkl = os.path.join('pkl','StopWords.pkl')
TweetDataPkl = os.path.join('pkl','TweetData.pkl')
FeaturesListPkl = os.path.join('pkl','FeaturesList.pkl')


   

allTweetData = []
allFeaturesList = getAllFeaturesList()


if (not os.path.exists('./%s'%TweetDataPkl) or not os.path.exists('./%s'%FeaturesListPkl)) :
    
    with open('positive-data.csv', 'r') as myopen:    
        reader = csv.reader(myopen, delimiter=',')
        for val in reader:
            feature_vector = getFeatureVector(val[0])
            allTweetData.append((feature_vector,'positive'))        
 
    with open('negative-data.csv', 'r') as myopen:    
        reader = csv.reader(myopen, delimiter=',')
        for val in reader:
            feature_vector = getFeatureVector(val[0])
            allTweetData.append((feature_vector,'negative')) 
         
    
    pickle.dump(allFeaturesList,open(FeaturesListPkl, 'wb'))
    pickle.dump(allTweetData,open(TweetDataPkl, 'wb'))
else:
    allFeaturesList = pickle.load(open(FeaturesListPkl,'rb')) 
    allTweetData = pickle.load(open(TweetDataPkl,'rb'))  



#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in allFeaturesList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


# Extract feature vector for all tweets in one shote
training_set = nltk.classify.util.apply_features(extract_word_features, allTweetData)  


def evaluate_classifier(data):
    
    trainfeats, testfeats  = train_test_split(data, test_size=0.3, random_state=0)
    
    # using 3 classifiers
    classifier_list = ['nb','svm']
    classifier_dict ={'nb':'Naive Bayes', 'svm':'SVM'}     
        
    for cl in classifier_list:
        classifierPkl = os.path.join('pkl',cl+".pkl")
        if not os.path.exists('./%s'%classifierPkl):
            if cl == 'svm':
                classifier = SklearnClassifier(LinearSVC(), sparse=False)
                classifier.train(trainfeats)
            else:
                classifier = NaiveBayesClassifier.train(trainfeats)
            pickle.dump(classifier,open(classifierPkl, 'wb'))
        else:                 
            classifier = pickle.load(open(classifierPkl,'rb'))    
                
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
 
        for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)
 
        accuracy = nltk.classify.util.accuracy(classifier, testfeats)
        pos_precision = precision(refsets['positive'], testsets['positive'])
        pos_recall = recall(refsets['positive'], testsets['positive'])
        pos_fmeasure = f_measure(refsets['positive'], testsets['positive'])
        neg_precision = precision(refsets['negative'], testsets['negative'])
        neg_recall = recall(refsets['negative'], testsets['negative'])
        neg_fmeasure =  f_measure(refsets['negative'], testsets['negative'])
        
        print('')
        print('---------------------------------------')
        print('SINGLE FOLD RESULT ' + '(' + classifier_dict[cl] + ')')
        print('---------------------------------------')
        print('accuracy:', accuracy)
        print('precision', (pos_precision + neg_precision) / 2)
        print('recall', (pos_recall + neg_recall) / 2)
        print('f-measure', (pos_fmeasure + neg_fmeasure) / 2)
                
        #classifier.show_most_informative_features()
    
    print('')

    
       
    n = 5 # 5-fold cross-validation    
    
    for cl in classifier_list:
        
        subset_size = len(trainfeats) / n
        accuracy = []
        pos_precision = []
        pos_recall = []
        neg_precision = []
        neg_recall = []
        pos_fmeasure = []
        neg_fmeasure = []
        cv_count = 1
        for i in range(n):        
            testing_this_round = trainfeats[i*int(subset_size):][:int(subset_size)]
            training_this_round = trainfeats[:i*int(subset_size)] + trainfeats[(i+1)*int(subset_size):]
            classifierPkl = os.path.join('pkl',cl+"_cv.pkl")
            if not os.path.exists('./%s'%classifierPkl):
                if cl == 'svm':
                    classifier = SklearnClassifier(LinearSVC(), sparse=False)
                    classifier.train(training_this_round)
                else:
                    classifier = NaiveBayesClassifier.train(training_this_round)
                pickle.dump(classifier,open(classifierPkl, 'wb'))         
            else:
                classifier = pickle.load(open(classifierPkl,'rb'))                           
            refsets = collections.defaultdict(set)
            testsets = collections.defaultdict(set)
            for i, (feats, label) in enumerate(testing_this_round):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)
            
            cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)
            cv_pos_precision = precision(refsets['positive'], testsets['positive'])
            cv_pos_recall = recall(refsets['positive'], testsets['positive'])
            cv_pos_fmeasure = f_measure(refsets['positive'], testsets['positive'])
            cv_neg_precision = precision(refsets['negative'], testsets['negative'])
            cv_neg_recall = recall(refsets['negative'], testsets['negative'])
            cv_neg_fmeasure =  f_measure(refsets['negative'], testsets['negative'])
                    
            accuracy.append(cv_accuracy)
            pos_precision.append(cv_pos_precision)
            pos_recall.append(cv_pos_recall)
            neg_precision.append(cv_neg_precision)
            neg_recall.append(cv_neg_recall)
            pos_fmeasure.append(cv_pos_fmeasure)
            neg_fmeasure.append(cv_neg_fmeasure)
            
            cv_count += 1
                
        print('---------------------------------------')
        print('N-FOLD CROSS VALIDATION RESULT ' + '(' + classifier_dict[cl] + ')')
        print('---------------------------------------')
        print('accuracy:', sum(accuracy) / n)
        print('precision', (sum(pos_precision)/n + sum(neg_precision)/n) / 2)
        print('recall', (sum(pos_recall)/n + sum(neg_recall)/n) / 2)
        print('f-measure', (sum(pos_fmeasure)/n + sum(neg_fmeasure)/n) / 2)
        print('')
    
        
evaluate_classifier(training_set)

 

  
               
        
        
        
        
        
        