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
from nltk.corpus import stopwords
# I bring 10 stopwords for the test observation from nltk stopwords data
print(stopwords.words('english')[:10])
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']

# I input what I processed to tokens and I remove stopwords in tokens by for iteration
words = [w for w in words if not w in stopwords.words('english')]
print(len(words))
# 219
print(words[:10])
# I can't see with, all like them
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

# I implement review_to_words() to process string sequence
def review_to_words(raw_review):
    # 1. I remove html tags in review text by using BeautifulSoup
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. I replace what not english character with a white space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. I convert all string to lower case character and I split them by word
    words = letters_only.lower().split()
    # 4. In, Python, it's faster to find by set than by list
    # I convert stopwords to set type
    stops = set(stopwords.words('english'))
    # 5. I remove stopwords
    meaningful_words = [w for w in words if not w in stops]
    # 6. I process stemmization
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. I bind all words which are delimited by white space to make string sequenc sentence as a return value
    return( ' '.join(stemming_words) )

# I use review_to_words() which is made just above
clean_review = review_to_words(train['review'][0])
print(clean_review)
# 'stuff go moment mj start listen music watch odd documentari watch wiz watch moonwalk mayb want get certain insight guy thought realli cool eighti mayb make mind whether guilti innoc moonwalk part biographi part featur film rememb go see cinema origin releas subtl messag mj feel toward press also obvious messag drug bad kay visual impress cours michael jackson unless remot like mj anyway go hate find bore may call mj egotist consent make movi mj fan would say made fan true realli nice actual featur film bit final start minut exclud smooth crimin sequenc joe pesci convinc psychopath power drug lord want mj dead bad beyond mj overheard plan nah joe pesci charact rant want peopl know suppli drug etc dunno mayb hate mj music lot cool thing like mj turn car robot whole speed demon sequenc also director must patienc saint came film kiddi bad sequenc usual director hate work one kid let alon whole bunch perform complex danc scene bottom line movi peopl like mj one level anoth think peopl stay away tri give wholesom messag iron mj bestest buddi movi girl michael jackson truli one talent peopl ever grace planet guilti well attent gave subject hmmm well know peopl differ behind close door know fact either extrem nice stupid guy one sickest liar hope latter'

# I process entire train data by the way I did before for the review of the first data
# I bring the entire review from train data
num_reviews = train['review'].size
print(num_reviews)
# 25000

"""
clean_train_reviews = []
In kaggle tutorial, range's written by xrange
But, for this case, since I use python3, I use range as following
"""
# for i in range(0, num_reviews):
#     clean_train_reviews.append( review_to_words(train['review'][i]))

"""
But, above code doesn't give information how far position the code is executing
So, I fix them to give information current state one time per 5000 unit
"""
# clean_train_reviews = []
# for i in range(0, num_reviews):
#     if (i + 1)%5000 == 0:
#         print('Review {} of {} '.format(i+1, num_reviews))
#     clean_train_reviews.append(review_to_words(train['review'][i]))
    
"""
To make the code brevity, I use apply instead of for loop
"""    
# %time train['review_clean'] = train['review'].apply(review_to_words)

"""
The code becomes simple but takes much of time
"""
# CPU times: user 1min 15s, sys: 2.3 s, total: 1min 18s
# Wall time: 1min 20s


from multiprocessing import Pool
import numpy as np

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    # 키워드 항목 중 workers 파라메터를 꺼냄
    workers = kwargs.pop('workers')
    # 위에서 가져온 workers 수로 프로세스 풀을 정의
    pool = Pool(processes=workers)
    # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    # 작업 결과를 합쳐서 반환
    return pd.concat(list(result))

def square(x):
    return x**x

if __name__ == '__main__':
    df = pd.DataFrame({'a':range(10), 'b':range(10)})
    apply_by_multiprocessing(df, square, axis=1, workers=4)  
    ## run by 4 processors

# clean_train_reviews = apply_by_multiprocessing(train['review'], review_to_words, workers=4)
clean_train_reviews = apply_by_multiprocessing(train['review'], review_to_words, workers=4)
print(clean_train_reviews)
# # CPU times: user 106 ms, sys: 119 ms, total: 226 ms
# # Wall time: 43.1 s

# clean_test_reviews = apply_by_multiprocessing(test['review'], review_to_words, workers=4)
clean_test_reviews = apply_by_multiprocessing(test['review'], review_to_words, workers=4)
print(clean_test_reviews)
# # CPU times: user 116 ms, sys: 139 ms, total: 255 ms
# # Wall time: 51.6 s

# I use "word cloud" showing visualization based on frequency of word
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def displayWordCloud(data = None, backgroundcolor = 'white', width=800, height=600 ):
    wordcloud = WordCloud(stopwords = STOPWORDS, 
                          background_color = backgroundcolor, 
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

# I draw "wordcloud" with all words of train data
displayWordCloud(' '.join(clean_train_reviews))




# I want to show visualization of data by seaborn
import seaborn as sns
# I will make 2 separated graphs(distributioin of the number of word per review, distributioin of the number of unique word per review without duplication of word)
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(18, 6)
print('the mean value of the number of words from each review:', train['num_words'].mean())
print('the median value of the number of words from each review:', train['num_words'].median())
# ax=axes[0] means column 0 out of 0,1
sns.distplot(train['num_words'], bins=100, ax=axes[0])
# vertical dot line representing median value
axes[0].axvline(train['num_words'].median(), linestyle='dashed')
axes[0].set_title('distributioin of the number of word per review')
# the mean value of the number of words from each review : 119.52356
# the median value of the number of words from each review : 89.0

print('the mean value of the number of unique words from each review:', train['num_uniq_words'].mean())
print('the median value of the number of unique words from each review:', train['num_uniq_words'].median())
sns.distplot(train['num_uniq_words'], bins=100, color='g', ax=axes[1])
axes[1].axvline(train['num_uniq_words'].median(), linestyle='dashed')
axes[1].set_title('distributioin of the number of unique word per review')
# the mean value of the number of unique words from each review : 94.05756
# the median value of the number of unique words from each review : 74.0




# Bag-of-words model - Wikipedia
# Let's suppose there are 2 sentences below
# (1) John likes to watch movies. Mary likes movies too.
# (2) John also likes to watch football games.

# I tokenize above 2 sentences, and put them into bags, then it seems like below
# [
#     "John",
#     "likes",
#     "to",
#     "watch",
#     "movies",
#     "Mary",
#     "too",
#     "also",
#     "football",
#     "games"
# ]

# And I count how many times each token shows up in the bag in order from top to bottom
# (1) John:1, likes:2, ... =>  [1, 2, 1, 1, 2, 1, 1, 0, 0, 0]
# (2) [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
# Above processing is the process of converting the formation of data to make the machine learning algorithm understand data

# I put "bag of words" in the "bigram" way by n-gram methodology, and it's shown below
# [
#     "John likes",
#     "likes to",
#     "to watch",
#     "watch movies",
#     "Mary likes",
#     "likes movies",
#     "movies too",
# ]
# I do above process by using CountVectorizer



# I generate features by using CountVectorizer of Scikit-learn
# I extract tokens by using regular expression
# Since I convert all alphabet characters to lowercase one, so that, all the words like "good, Good, gOod" will be identical by "good"
# Since this process generates meaningless features, so, I just only use tokens which are shown in at least 2 reviews
# I can define the minimal number of review which can define meanful token


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# I manipulate some parameter values, compared to tutorial
# Manipulating some parameter values can differ score of result
# I bring CountVectorizer
vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 2, # This is minimal number of document which can define meaningful token
                             ngram_range=(1, 3),
                             max_features = 20000
                            )
print(vectorizer)
# CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#         dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#         lowercase=True, max_df=1.0, max_features=20000, min_df=2,
#         ngram_range=(1, 3), preprocessor=None, stop_words=None,
#         strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
#         tokenizer=None, vocabulary=None)

# I use "pipeline" to enhance processing speed
# Reference : https://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph
# I bring "pipeline" with passing vectorizer which I made above
pipeline = Pipeline([
    ('vect', vectorizer),
])
# I process clean_train_reviews by pipeline.fit_transform()
train_data_features = pipeline.fit_transform(clean_train_reviews)
print(train_data_features)
# <25000x20000 sparse matrix of type '<class 'numpy.int64'>'
# 	with 2762268 stored elements in Compressed Sparse Row format>

print(train_data_features.shape)
# (25000, 20000)

vocab = vectorizer.get_feature_names()
print(len(vocab))
# 20000
vocab[:10]
# ['aag',
#  'aaron',
#  'ab',
#  'abandon',
#  'abbey',
#  'abbi',
#  'abbot',
#  'abbott',
#  'abc',
#  'abduct']

# I check out vectorized feature
import numpy as np
dist = np.sum(train_data_features, axis=0)
    
for tag, count in zip(vocab, dist):
    print(count, tag)
    
pd.DataFrame(dist, columns=vocab)
# [[26 48 22 ... 59 40 23]] aag

# aag	aaron	ab	abandon	abbey	abbi	abbot	abbott	abc	abduct	...	zombi bloodbath	zombi film	zombi flick	zombi movi	zone	zoo	zoom	zorro	zu	zucker
# 0	26	48	22	288	24	30	29	30	125	55	...	23	52	37	89	161	31	71	59	40	23
# 1 rows × 20000 columns

# I bring 10 train_data_features from the first location, and show their header
pd.DataFrame(train_data_features[:10].toarray(), columns=vocab).head()
# aag	aaron	ab	abandon	abbey	abbi	abbot	abbott	abc	abduct	...	zombi bloodbath	zombi film	zombi flick	zombi movi	zone	zoo	zoom	zorro	zu	zucker
# 0	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
# 1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
# 2	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
# 3	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
# 4	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
# 5 rows × 20000 columns


# I use random forest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1, random_state=2018)
print(forest)

# I set up random forest classifier in detail
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
            oob_score=False, random_state=2018, verbose=0,
            warm_start=False)

# I input feature(train_data_features) and label(train['sentiment']) into random forest classifier to make it learn(fit)
forest = forest.fit(train_data_features, train['sentiment'])
# CPU times: user 1min 16s, sys: 324 ms, total: 1min 17s
# Wall time: 20.9 s

# I use cross validation
from sklearn.model_selection import cross_val_score

# I input data and get result, then I find mean value from the result 
score = np.mean(cross_val_score(forest, train_data_features, train['sentiment'], cv=10, scoring='roc_auc'))
# CPU times: user 10min 52s, sys: 3.19 s, total: 10min 55s
# Wall time: 2min 57s

# I check out the first test data which I refined above
print(clean_test_reviews[0])
# 'natur film main theme mortal nostalgia loss innoc perhap surpris rate high older viewer younger one howev craftsmanship complet film anyon enjoy pace steadi constant charact full engag relationship interact natur show need flood tear show emot scream show fear shout show disput violenc show anger natur joyc short stori lend film readi made structur perfect polish diamond small chang huston make inclus poem fit neat truli masterpiec tact subtleti overwhelm beauti'

# I vectorize test data as in the same way I did for train data
test_data_features = pipeline.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
# CPU times: user 8.18 s, sys: 46.6 ms, total: 8.23 s
# Wall time: 8.24 s
print(test_data_features)
# I can see sparse matrix which is made from vectorized words
# array([[0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        ...,
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0],
#        [0, 0, 0, ..., 0, 0, 0]])

# I count the number of vectorized word of how many times they show up in the reviews
# I bring 5th data with its data up to 100 from the first location
print(test_data_features[5][:100])
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# I can find and see what the word is
# vocab = vectorizer.get_feature_names()
print(vocab[8], vocab[2558], vocab[2559], vocab[2560])
# ('abc', 'charact person', 'charact play', 'charact plot')

# I input test data into random forest and make it to predict label value
result = forest.predict(test_data_features)
print(result[:10])
# array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0])

# I store pridicted values into dataframe
output = pd.DataFrame(data={'id':test['id'], 'sentiment':result})
print(output.head())
# id	sentiment
# 0	"12311_10"	1
# 1	"8348_2"	0
# 2	"5828_4"	0
# 3	"7186_2"	1
# 4	"12128_7"	1

output.to_csv('D://chromedown//tutorial_1_BOW_{0:.5f}.csv'.format(score), index=False, quoting=3)

output_sentiment = output['sentiment'].value_counts()
print(output_sentiment[0] - output_sentiment[1])
# 108
print(output_sentiment)
# 0    12554
# 1    12446
# Name: sentiment, dtype: int64

# I draw graphs to compare train ones and test output ones of sentiment
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.countplot(train['sentiment'], ax=axes[0])
sns.countplot(output['sentiment'], ax=axes[1])
