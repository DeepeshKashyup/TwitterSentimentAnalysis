#from nltk.tokenize import sent_tokenize, word_tokenize

#EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

#print(sent_tokenize(EXAMPLE_TEXT))

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer


## Stop words
stop_words= set(stopwords.words('english'))

example_sent ='I stayed here in this apartment and had pretty bad experience with these guys. Here is my 4 reasons why you should not choose this apartment 1) No apartment in Bentonville, AR will collect initial amount of $700 as caution deposit. You will realize later why Waterside apartments collecting $700.'


word_tokenized = word_tokenize(example_sent)

filtered_sentence= []


for w in word_tokenized:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)


## Stemming

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in example_words:
    print(ps.stem(w))


for w in filtered_sentence:
    print(ps.stem(w))

## part of speech tagging

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")


custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

#print tokenized

def process_content():
    try:
        for i in tokenized[:5]:
            words= nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()
        
