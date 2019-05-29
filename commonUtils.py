import re
import os
import pickle


#put Name of all Serialized objects
SentimentDictPkl = os.path.join('pkl','SentimentDict.pkl')
PositiveFeaturesPkl = os.path.join('pkl','PositiveFeatures.pkl')
NegativeFeaturesPkl = os.path.join('pkl','NegativeFeatures.pkl')
StopWordsPkl = os.path.join('pkl','StopWords.pkl')
FeaturesListPkl = os.path.join('pkl','FeaturesList.pkl')

lexiconopenName = 'lexicon.txt'
stopWordsopenName = 'stopwords.txt'

#start getStopWordList
def getStopWordList():
    if not os.path.exists('./%s'%StopWordsPkl):       
        #read the stopwords open and build a list
        stopWords = []
        stopWords.append('AT_USER')
        stopWords.append('URL')
    
        fp = open(stopWordsopenName, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stopWords.append(word)
            line = fp.readline()
        fp.close()
        pickle.dump(stopWords,open(StopWordsPkl, 'wb'))
    else:
        stopWords = pickle.load(open(StopWordsPkl,'rb'))
    return stopWords


#This method will return sentiment dictionary of postive and negative words
def getSentimentDictionary():
    if not os.path.exists('./%s'%SentimentDictPkl):
        sentiment_dictionary = {}
        lexicon = open(lexiconopenName,'r');
        for line in  lexicon.read().split('\r'):
            if line.endswith('positive'):
                positive_word = line.split('\t')[0]; 
                sentiment_dictionary[positive_word] = 1;
            elif line.endswith('negative'):
                negative_word = line.split('\t')[0]; 
                sentiment_dictionary[negative_word] = -1;
        pickle.dump(sentiment_dictionary,open(SentimentDictPkl, 'wb'))
    else:
        sentiment_dictionary = pickle.load(open(SentimentDictPkl,'rb'))
        
    return sentiment_dictionary 

def getPositiveFeatures():
    if not os.path.exists('./%s'%PositiveFeaturesPkl):
        positive_features = []
        lexicon = open(lexiconopenName,'r');
        for line in  lexicon.read().split('\r'):
            if line.endswith('positive'):
                positive_word = line.split('\t')[0]; 
                positive_features.append(positive_word);               
        pickle.dump(positive_features,open(PositiveFeaturesPkl, 'wb'))
    else:
        positive_features = pickle.load(open(PositiveFeaturesPkl,'rb'))
        
    return positive_features        

def getNegativeFeatures():
    if not os.path.exists('./%s'%NegativeFeaturesPkl):
        negative_features = []
        lexicon = open(lexiconopenName,'r');
        for line in  lexicon.read().split('\r'):
            if line.endswith('negative'):
                negative_word = line.split('\t')[0]; 
                negative_features.append(negative_word);               
        pickle.dump(negative_features,open(NegativeFeaturesPkl, 'wb'))
    else:
        negative_features = pickle.load(open(NegativeFeaturesPkl,'rb'))
        
    return negative_features 


def getAllFeaturesList():
    if not os.path.exists('./%s'%FeaturesListPkl):
        allFeaturesList = []  
        allFeaturesList.extend(getPositiveFeatures())
        allFeaturesList.extend(getNegativeFeatures())      
        #remove duplicates from feature List
        allFeaturesList = list(set(allFeaturesList));         
        pickle.dump(allFeaturesList,open(FeaturesListPkl, 'wb'))
    else:
        allFeaturesList = pickle.load(open(FeaturesListPkl,'rb'))
        
    return allFeaturesList


# start process tweet
def processTweet(tweet):
    #convert to lower case
    tweet = tweet.lower();
    
    ##Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end
   
