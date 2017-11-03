# Natural Language Processing with NLTK 

NLTK module is a massive toolkit, aimed at helping you wit the entire Natual Language Processing(NLP) methodology.
NLTK will aid you with everything from splitting sentences from paragraphs, splitting up words, recognizing the part of speech of those words, highlighting the main subject, and then
even helping your machine to undertstand what the text is all about. 

Here, we are going to tackle the field of opinion mining, or sentiment analysis.

This project covers the following:

Tokenizing - Splitting Sentences and words from the body of text

Part of speech tagging

Machine learning with the Naive Bayes Classfier

Training Classifiers with Datasets

Performing live, streaming, sentiment analysis with twitter


## vocabulary:

**Corpus** - Body of text, singular. Corpora is the plural of this. Example: A collection of medical journals.

**Lexicon** - Words and their meanings. Example: English dictionary. Consider, however, that various fields will have different lexicons. For example: To a financial investor, the first meaning for the word "Bull" is someone who is confident about the market, as compared to the common English lexicon, where the first meaning for the word "Bull" is an animal. As such, there is a special lexicon for financial investors, doctors, children, mechanics, and so on.

**Token** - Each "entity" that is a part of whatever was split up based on rules. For examples, each word is a token when a sentence is "tokenized" into words. Each sentence can also be a token, if you tokenized the sentences out of a paragraph.

**Stop Words** - words that just contain no meaning, and we want to remove them.

## Processing

**Stop Words Removal**

**Stemming** - The idea of stemming is a sort of normalizing method. Many variations of words carry the same meaning, other than when tense is involved.
The reason why we stem is to shorten the lookup, and normalize sentences.

**Speech Tagging** - Labelling the words in a sentence as nouns, adjectives, verbs ...etc. Even more impressive, it also labels by tense, and more

Here is the POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when


**Chunking** -	Grouping of words into hopefully meaningful chunks. One of the main goals of chunking is to group into what are known as "noun phrases." These are phrases of one or more words that contain a noun, maybe some descriptive words, maybe a verb, and maybe something like an adverb. The idea is to group nouns with the words that are in relation to them.

**Chinking** - Chinking is a lot like chunking, it is basically a way for you to remove a chunk from a chunk. The chunk that you remove from your chunk is your chink.

**NER** - One of the most major forms of chunking in natural language processing is called "Named Entity Recognition." The idea is to have the machine immediately be able to pull out "entities" like people, places, things, locations, monetary figures, and more.

**Lemmatizing** - A very similar operation to stemming is called lemmatizing. The major difference between these is, as you saw earlier, stemming can often create non-existent words, whereas lemmas are actual words.So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma.

**Text Classfication** - The goal with text classification can be pretty broad. Maybe we're trying to classify text as about politics or the military. Maybe we're trying to classify it by the gender of the author who wrote it. A fairly popular text classification task is to identify a body of text as either spam or not spam, for things like email filters. 
 
**Word to Feature Conversion** - compiling feature lists of words from positive reviews and words from the negative reviews to hopefully see trends in specific types of words in positive or negative reviews.

## NLTK Corpus

The NLTK corpus is a massive dump of all kinds of natural language data sets that are definitely worth taking a look at.

Almost all of the files in the NLTK corpus follow the same rules for accessing them by using the NLTK module, but nothing is magical about them. These files are plain text files for the most part, some are XML and some are other formats, but they are all accessible by you manually, or via the module and Python. Let's talk about viewing them manually.

