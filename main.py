from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import matplotlib
from nltk import FreqDist

from nltk.stem import SnowballStemmer
from nltk.book import *
from nltk.stem import WordNetLemmatizer
import numpy
import nltk

#nltk.download("book")
# nltk.download('punkt')
# nltk.download()
# print(nltk.data.path)
# nltk.download("maxent_ne_chunker")
# nltk.download("words")

#Technique 1: Tokenizing
example_string = """
... Muad'Dib learned rapidly because his first training was in how to learn.
... And the first lesson of all was the basic trust that he could learn.
... It's shocking to find how many people do not believe they can learn,
... and how many more believe learning to be difficult."""

# split into sentences
sentences = sent_tokenize(example_string)

# split into words
words = word_tokenize(example_string)


#Technique 2: Stop Words
example_quote = "Apple is a very good company and their stock price is increasing every day. Other companies"
words_in_quote = word_tokenize(example_quote)

# create set of stopwords
stop_words = set(stopwords.words("english"))

# emmpty list für das Ergebnis
filtered_list = []

for word in words_in_quote:
    if word.casefold() not in stop_words:
        filtered_list.append(word)

print(filtered_list)

#Technique 3:  Stemming
# Instanz von PorterStemmer
snow_stemmer = SnowballStemmer("english")

string_for_stemming = """
... The crew of the USS Discovery discovered many discoveries.
... Discovering is what explorers do."""
words = word_tokenize(string_for_stemming)
stemmed_words = []

for word in words:
    stemmed_words.append(snow_stemmer.stem(word))

print(stemmed_words)

#Technique 4: Parts of Speech
print(nltk.pos_tag(words))

#Technique 5: Lemmantizier
# Objectinstantzierung
quote = "My friends love telling me stories about scarves."
words_in_quote = word_tokenize(quote)

lemmatizer = WordNetLemmatizer()
lemmas = []
for word in words_in_quote:
    lemmas.append(lemmatizer.lemmatize(word))

print(lemmas)

print(lemmatizer.lemmatize("worst"))
print(lemmatizer.lemmatize("worst", pos='a'))

#Technique 6: Chunking
lotr_quote = "It's a dangerous business, Frodo, going out your door."
lotr_words = word_tokenize(lotr_quote)
lotr_pos = nltk.pos_tag(lotr_words)
print(lotr_pos)
grammar = "NP: {<DT>?<JJ>*<NN>}"

chunk_parser = nltk.RegexpParser(grammar)

tree = chunk_parser.parse(lotr_pos)
tree.draw()

#Technique 7: Chinking
grammar = """
Chunk: {<.*>+}
       }<JJ>{"""
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(lotr_pos)
tree.draw()

#Technique 8: Named Entity Recognition
tree = nltk.ne_chunk(lotr_pos)
tree.draw()

#Methode zur Extraktion von NE
def extract_ne(quote):
    words = word_tokenize(quote)
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )

#Technique 9: Frequency
meaningfull_words = [
    word for word in text8 if word.casefold() not in stop_words
]
frequencydist = FreqDist(meaningfull_words)
frequencydist.plot(20, cumulative=True)

#Dispersion
text8.dispersion_plot(
    ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
)

#Collocations
lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]
new_text = nltk.Text(lemmatized_words)
