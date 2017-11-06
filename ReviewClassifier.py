import nltk
from nltk.tokenize import word_tokenize
from io import open
import pickle
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC



pos_rev = open("positive.txt","r").read()

neg_rev = open("negative.txt","r").read()


# j is adjective, R is adverb, and V is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]


##build documents (rev,category)

document = []

##print len(document)

##build a Wordlist of most commonly used words top 3000 and filter them for adjectives

all_words = []

for line in pos_rev.split('\n'):
    document.append((line,"pos"))
##    print line
    words = word_tokenize(line)
##    print words
    pos = nltk.pos_tag(words)
#    print pos
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



for line in neg_rev.split('\n'):
    document.append((line,"neg"))
    words = word_tokenize(line)
    pos= nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())



save_document = open("pickled_algos/documents.pickle","wb")
pickle.dump(document,save_document)
save_document.close()

all_words = nltk.FreqDist(all_words)

##build a featureSet of features with category

word_features =  list(all_words)[:5000]


save_word_feature = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features,save_word_feature)
save_word_feature.close()

def find_features(document):
    words = set(nltk.word_tokenize(document))
    features = {}
    pos= nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            features[w[0]] = (w[0] in word_features)
    return features

featureSet = [(find_features(doc),cat) for doc,cat in document]

random.shuffle(featureSet)

print len(featureSet)
##form test and train data from features

training_set = featureSet[:10000]
testing_set = featureSet[10000:]

##train classfier on train data set
classifier = nltk.NaiveBayesClassifier.train(training_set)

##check accuracy of classifier
print "Original Naive Bayes Algo accuracy percent",(nltk.classify.accuracy(classifier,testing_set)*100)

print classifier.show_most_informative_features(15)


###############
save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close
