import os
import pickle
from sentimentUtils import getFeatureVector,extract_word_features


inputDirName = 'Twitter Data';
outputDirName = 'Result'

#Sentiment prediction using Best Classifier model saved using pickle
# ---------------------------------------
# SINGLE FOLD RESULT (Naive Bayes)
# ---------------------------------------
# accuracy: 0.83
# precision 0.853936545241
# recall 0.835202991453
# f-measure 0.828396460256

# ---------------------------------------
# SINGLE FOLD RESULT (SVM)
# ---------------------------------------
# accuracy: 0.813333333333
# precision 0.813333333333
# recall 0.813835470085
# f-measure 0.813258636788

# ---------------------------------------
# N-FOLD CROSS VALIDATION RESULT (Naive Bayes)
# ---------------------------------------
# accuracy: 0.948571428571
# precision 0.950264364014
# recall 0.946820784157
# f-measure 0.94716754863

# ---------------------------------------
# N-FOLD CROSS VALIDATION RESULT (SVM)
# ---------------------------------------
# accuracy: 0.951428571429
# precision 0.950980392157
# recall 0.952481076535
# f-measure 0.950937950938
# ---------------------------------------

BestClassifierPkl = os.path.join("pkl","nb_cv.pkl");
BestClassifier = pickle.load(open(BestClassifierPkl,'rb'))
    

FeaturesListPkl = os.path.join('pkl','FeaturesList.pkl')
allFeaturesList = pickle.load(open(FeaturesListPkl,'rb')) 

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in allFeaturesList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end



feedsopenList = []
for openname in os.listdir(inputDirName):
        feedsopenList.append(openname)

if not os.path.exists(outputDirName):
        os.makedirs(outputDirName) 
for feedsopenName in feedsopenList: 
    print("started processing open ",feedsopenName )
    fullInputopenName = os.path.join(inputDirName,feedsopenName);       
    feeds_input_open = open(fullInputopenName, 'r', encoding="utf8", errors='ignore');
    sentiment_result = [] 
    fullOutputopenName = os.path.join(outputDirName,feedsopenName);
    feeds_output_open = open(fullOutputopenName, 'w+', encoding="utf8", errors='ignore');

    for tweet in  feeds_input_open.read().split('\r'):  
        if(len(tweet) > 0):           
            feature_vector = getFeatureVector(tweet);
            extracted_features = extract_word_features(feature_vector)
            sentiment = BestClassifier.classify(extracted_features) 
            sentiment_result.append(tweet+" :: "+sentiment+"\r");           
     
    feeds_output_open.writelines(sentiment_result);
    print("finished predicting sentiment for open ",feedsopenName )
    feeds_input_open.close();
    feeds_output_open.close();         
     
    