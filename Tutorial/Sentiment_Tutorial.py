import deepcut
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_validate, cross_val_predict

def feature_vectoriser (sentence, feature_dictionary) :

    features = [0] * len(feature_dictionary)
    for word in sentence :
        if word in feature_dictionary.keys() :
            features[feature_dictionary[word]] += 1
    return features

def get_list_of_word (sentence) :
    return deepcut.tokenize(sentence)

def get_feature_list (sentence_list) :
    features = set()
    for word_list,sentiment in sentence_list :
        features = features.union(word_list)
    
    feature_dictionary = dict()
    i = 0
    for word in features :
        feature_dictionary[word] = i
        i = i+1

    print (str(len(feature_dictionary)) + ' features has been exported')
    return feature_dictionary

# Load Data

neg_text = [(line.strip(), '1') for line in open("data/neg.txt", 'r')]
neutral_text = [(line.strip(), '2') for line in open("data/neutral.txt", 'r')]
pos_text = [(line.strip(), '3') for line in open("data/pos.txt", 'r')]

# Combine All Data Together
full_data = [(get_list_of_word(sentence), sentiment) for (sentence, sentiment) in neg_text + neutral_text + pos_text]

# Build Feature Space
feature_list = get_feature_list(full_data)

X = []
y = []

for words, sentiment in full_data : 
    X.append(feature_vectoriser(words, feature_list))
    y.append(sentiment)

# K-Fold Test
model = BernoulliNB()
result_matric = cross_validate(model, X, y, cv=10, n_jobs=-1, scoring=['precision_macro', 'recall_macro'])
print ("Precision " + str(round(np.array(result_matric['test_precision_macro']).mean()*100,2)) + "%")
print ("Recall " + str(round(np.array(result_matric['test_recall_macro']).mean()*100,2)) + "%")
