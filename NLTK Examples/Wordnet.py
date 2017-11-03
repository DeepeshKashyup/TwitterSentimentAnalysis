

from nltk.corpus import wordnet

syns = wordnet.synsets("God")

#example of synset
print syns[0].name()

#getting just the word
for syn in syns:
    print syn.lemmas()[0].name()


##defination of the first synset


print(syns[0].definition())

##Example of the word in use:

print(syns[0].examples())

synonyms = []
antonyms = []

for word in wordnet.synsets("Good"):
    for l in word.lemmas():
##        print l.name()
        w=l.name()
        synonyms.append(w)
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print wordnet.synsets("Goodness")[0].lemmas()[0].antonyms()
print(synonyms)

print(antonyms)


##WordNet to compare the similarity of two words and their tenses,
##by incorporating the Wu and Palmer method for semantic related-ness.


w1 = wordnet.synset("cauliflower.n.01")
w2 = wordnet.synset("broccoli.n.01")

print w1.wup_similarity(w2)


w1 = wordnet.synset("cauliflower.n.01")
w2 = wordnet.synset("mango.n.01")

print w1.wup_similarity(w2)



w1 = wordnet.synset("cauliflower.n.01")
w2 = wordnet.synset("table.n.01")

print w1.wup_similarity(w2)
