import nltk
import random

from nltk.corpus import movie_reviews

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

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

#set we will use to train our classfier with
training_set = featureSets[:1900]


#set we will use to test our classifier
testing_set = featureSets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

#Test the accuracy of the classifier with test set
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
