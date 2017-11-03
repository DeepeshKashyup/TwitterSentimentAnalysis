import nltk
import random
from nltk.corpus import movie_reviews

document= [(list(movie_reviews.words(fileid)),category)
       for category in movie_reviews.categories()
       for fileid in movie_reviews.fileids(category)]

random.shuffle(document)


##print document[1]



all_words= []

for w in movie_reviews.words():
    all_words.append(w.lower())


all_words = nltk.FreqDist(all_words)

# Word features contains 3000 most common words in movie reviews
word_features = all_words.keys()[:3000]

#print word_features


#function to find these common words in the positive or negative reviews and marking
#their presence as either positive or negative

def find_features(document):
    words = set(document)
    features ={}
    for  w in word_features:
        features[w] = (w in words)

    return features


#print find_features(movie_reviews.words('neg/cv000_29416.txt'))

featureSets = [(find_features(rev),category) for (rev,category) in document]

#print featureSets[1]

def filter_feature(features):
    f = {}
    for key in features.keys():
        if(features[key]):
            f[key] = True
    return f

filteredSet = [(filter_feature(features),category) for (features,category) in featureSets]

for f in filteredSet[:5]:
    print f
            
            
        
