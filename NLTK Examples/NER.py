import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_txt = state_union.raw("2005-GWBush.txt")
sample_txt = state_union.raw("2006-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)

tokenized = custom_sent_tokenizer.tokenize(sample_txt)

for i in tokenized:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    namedEnt = nltk.ne_chunk(tagged,binary=True)
    namedEnt.draw()
