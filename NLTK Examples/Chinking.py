import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union

train_txt= state_union.raw("2005-GWBush.txt")
sample_txt = state_union.raw("2006-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)

tokenized = custom_sent_tokenizer.tokenize(sample_txt)

print tokenized[:5]


for i in tokenized:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
##    print tagged
    chinkGram = r"""chink:{<.*>+}
                    }<VB.?|IN|DT|TO>+{"""
    chinkParser = nltk.RegexpParser(chinkGram)
    chinked = chinkParser.parse(tagged)
    chinked.draw()
