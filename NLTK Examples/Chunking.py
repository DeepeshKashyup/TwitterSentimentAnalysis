import nltk

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.draw import tree

train_txt = state_union.raw("2005-GWBush.txt")
sample_txt = state_union.raw("2006-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)
tokenized = custom_sent_tokenizer.tokenize(sample_txt)

#print tokenized[:2]


for i in tokenized[:5]:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
##    print tagged
    chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
##    print chunked.leaves()
##    chunked.draw()
##    print chunked
    for subtree in chunked.subtrees(filter=lambda t: t.label()=='Chunk'):
        print subtree


