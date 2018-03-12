# sampleSubmission.csv : sample for submission
# labeledTraindata.tsv : included evaluation either negative or positive(id-review-sentiment)
# testData.tsv : data set without sentiments(id-review)
# unlabeledTrainData.tsv

import pandas as pd
"""
header = 0 means that the first sentence is the name of columns
delimiter = \t means that elements are delimited by tab
quoting = 3 means that double quatation is neglected
# QUOTE_MINIMAL (0), 
# QUOTE_ALL (1), 
# QUOTE_NONNUMERIC (2),
# QUOTE_NONE (3).
"""

# I load labeledTrainData file which is composed of id-review-sentiment
train = pd.read_csv('D://chromedown//labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
# I load testData file which is composed of id-review
test = pd.read_csv('D://chromedown//testData.tsv', header=0, delimiter='\t', quoting=3)
print(train.shape)
# row, columns(id, sentiment, review) (25000, 3)
print(train.tail(3))
# I bring 3 data from at the end of train data set
# id	sentiment	review
# 24997	"10905_3"	0	"Guy is a loser. Can't get girls, needs to bui...
# 24998	"10194_3"	0	"This 30 minute documentary Buñuel made in the...
# 24999	"8478_8"	1	"I saw this movie as a child and it broke my h...
print(test.shape)
# (25000, 2)
print(test.tail())
# 	id	review
# 24995	"2155_10"	"Sony Pictures Classics, I'm looking at you! S...
# 24996	"59_10"	"I always felt that Ms. Merkerson had never go...
# 24997	"2531_1"	"I was so disappointed in this movie. I am ver...
# 24998	"7772_8"	"From the opening sequence, filled with black ...
# 24999	"11465_10"	"This is a great horror film for people who do...

# I show the names of columns of train data
print(train.columns.values)
# array(['id', 'sentiment', 'review'], dtype=object)

# I can see there is no sentiment data in test data
# I will expect sentiments of these data by machine learning
print(test.columns.values)
# array(['id', 'review'], dtype=object)

print(train.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 25000 entries, 0 to 24999
# Data columns (total 3 columns):
# id           25000 non-null object
# sentiment    25000 non-null int64
# review       25000 non-null object
# dtypes: int64(1), object(2)
# memory usage: 586.0+ KB

print(train.describe())
# sentiment
# count	25000.00000
# mean	0.50000
# std	0.50001
# min	0.00000
# 25%	0.00000
# 50%	0.50000
# 75%	1.00000
# max	1.00000

print(train['sentiment'].value_counts())
# 1    12500
# 0    12500
# Name: sentiment, dtype: int64

# I need to remove html tags in review data
print(train['review'][0][:700])
# I bring the review data of first data only with 700 words
# '"With all this stuff going down at the moment with MJ i\'ve started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ\'s feeling towards the press and also the obvious message of drugs are bad m\'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely lik'


from bs4 import BeautifulSoup
# I bring the review of the first data as html5 
example1 = BeautifulSoup(train['review'][0], "html5lib")
# I bring the review of the first data as html5 
print(train['review'][0][:700])
# "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely lik

# I can remove <br/> tags by doing the following processing
print(example1.get_text()[:700])
# '"With all this stuff going down at the moment with MJ i\'ve started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ\'s feeling towards the press and also the obvious message of drugs are bad m\'kay.Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyw'

import re
# I remove special characters by using regular expression
# I replace what not lower and upper case alphabet ([^a-zA-Z]) with white space (' ')
letters_only = re.sub('[^a-zA-Z]', ' ', example1.get_text())
print(letters_only[:700])
# ' With all this stuff going down at the moment with MJ i ve started listening to his music  watching the odd documentary here and there  watched The Wiz and watched Moonwalker again  Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent  Moonwalker is part biography  part feature film which i remember going to see at the cinema when it was originally released  Some of it has subtle messages about MJ s feeling towards the press and also the obvious message of drugs are bad m kay Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyw'

# I replace what not lower case alphabet with lower case alphabet
lower_case = letters_only.lower()

# I split all sentence into words, which is called tokenization
words = lower_case.split()
print(len(words))
# 437
print(words[:10])
# ['with',
#  'all',
#  'this',
#  'stuff',
#  'going',
#  'down',
#  'at',
#  'the',
#  'moment',
#  'with']

import nltk
nltk.download()
from nltk.corpus import stopwords
# I bring 10 stop words for the test observation from nltk stopwords data
print(stopwords.words('english')[:10])
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']

# I input what I processed to tokens and I remove stopwords in tokens by for iteration
words = [w for w in words if not w in stopwords.words('english')]
print(len(words))
# 219
print(words[:10])
# ['stuff',
#  'going',
#  'moment',
#  'mj',
#  'started',
#  'listening',
#  'music',
#  'watching',
#  'odd',
#  'documentary']

# This is an example of using PorterStemmer
stemmer = nltk.stem.PorterStemmer()
# I want to stemmize the word "maximum"
print(stemmer.stem('maximum'))
# maximum

# They are other examples
print("The stemmed form of running is: {}".format(stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(stemmer.stem("run")))
# The stemmed form of running is: run
# The stemmed form of runs is: run
# The stemmed form of run is: run

# This is an example of using LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('maximum'))
# maxim

print("The stemmed form of running is: {}".format(lancaster_stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(lancaster_stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(lancaster_stemmer.stem("run")))
# The stemmed form of running is: run
# The stemmed form of runs is: run
# The stemmed form of run is: run

# They are words before processing
print(words[:10])
# ['stuff',
#  'going',
#  'moment',
#  'mj',
#  'started',
#  'listening',
#  'music',
#  'watching',
#  'odd',
#  'documentary']

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
words = [stemmer.stem(w) for w in words]
# They are words after processing
print(words[:10])
# ['stuff',
#  'go',
#  'moment',
#  'mj',
#  'start',
#  'listen',
#  'music',
#  'watch',
#  'odd',
#  'documentari']

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

print(wordnet_lemmatizer.lemmatize('fly'))
print(wordnet_lemmatizer.lemmatize('flies'))
# fly
# fly

words = [wordnet_lemmatizer.lemmatize(w) for w in words]
# They are words after processing
print(words[:10])
# ['stuff',
#  'go',
#  'moment',
#  'mj',
#  'start',
#  'listen',
#  'music',
#  'watch',
#  'odd',
#  'documentari']



