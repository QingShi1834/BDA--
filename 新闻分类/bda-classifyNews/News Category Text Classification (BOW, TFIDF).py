import nltk
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib as plt
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_json('News_Category.json', lines=True)
print(df.head())

print(df.shape)

print('Number of Categories',df.groupby('category').ngroups)

df.groupby('category').size().sort_values(ascending=False)

df.groupby('category').size().sort_values(ascending=False).plot.bar()

df.groupby('category').size().sort_values(ascending=False)

categories = df.groupby('category').size().sort_values(ascending=False).reset_index(name='count')
df = df[ df['category'].isin(categories.iloc[:10,0])]

set(df['category'])

# Lets join the headline with the description, separated by a space, since both of them will contain useful information.

df['fulltext'] = df['headline'] + ' ' + df['short_description']

print(df.iloc[0,-1])


# Create column for number of words in each text

df['count_words'] = df['fulltext'].str.split().str.len()


df[['category','count_words']].groupby('category')['count_words'].median().plot.bar(y='count_words')


# ## Train Test Split
# I will do the train/test split before I do anything further since I need to have a consistent train and test set between all my experiments.

# Convert to Lowercase
df['fulltext_processed'] = df['fulltext'].str.lower()

X_train, X_test, y_train, y_test = train_test_split(df['fulltext_processed'], df['category'], test_size=0.2, random_state=0)

print("this is test :",X_test)
# print(X_train)
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn import preprocessing





# Maximum 1500 features and the words must have appeared in at least 10 texts
vectorizer1 = CountVectorizer(max_features=1500,min_df=5,max_df=0.7, stop_words=stopwords.words('english'))
X_train_counts = vectorizer1.fit_transform(X_train)

X_test_counts = vectorizer1.transform(X_test)

print(vectorizer1.get_feature_names())
print(X_train_counts.shape)
print(X_test_counts.shape)

# For each article we have 3000 columns, a count for each word
print(X_train_counts.toarray())

classifier = MultinomialNB().fit(X_train_counts, y_train)
y_pred = classifier.predict(X_test_counts)

print('confusion matrix:',confusion_matrix(y_test,y_pred))
print('classification report:', classification_report(y_test,y_pred))
print('accuracy:',accuracy_score(y_test, y_pred))


# Using a basic Bag of Words, uni-gram model, we get 69% accuracy. That's not bad, if we guessed everything to be politics only, then accuracy would've been 16%.

# ## TF-IDF
tfidf_converter = TfidfVectorizer(max_features=1500,min_df=5,max_df=0.7, stop_words=stopwords.words('english'))
X_train_tfidf = tfidf_converter.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_converter.transform(X_test).toarray() # don't use fit_transform, since we only want to fit on training set, but use the same vectoriser transform on the test


# Naive Bayes

classifier = MultinomialNB().fit(X_train_tfidf, y_train)
y_pred = classifier.predict(X_test_tfidf)
print('confusion matrix:',confusion_matrix(y_test,y_pred))
print('classification report:', classification_report(y_test,y_pred))
print('accuracy:',accuracy_score(y_test, y_pred))


# SVM
classifier = LinearSVC().fit(X_train_tfidf, y_train)
y_pred = classifier.predict(X_test_tfidf)
print('confusion matrix:',confusion_matrix(y_test,y_pred))
print('classification report:', classification_report(y_test,y_pred))
print('accuracy:',accuracy_score(y_test, y_pred))


# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', max_iter=100, C=1)
classifier = lr.fit(X_train_tfidf, y_train)
y_pred = classifier.predict(X_test_tfidf)
print('confusion matrix:',confusion_matrix(y_test,y_pred))
print('classification report:', classification_report(y_test,y_pred))
print('accuracy:',accuracy_score(y_test, y_pred))


# ## Stemming and Lemmatization

# Stemming and tokenising and remove stopwords in same function
from nltk.stem import PorterStemmer, WordNetLemmatizer
# lemmetizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def stem_text(rawsentence):
    stemmer = PorterStemmer()
    tokens = word_tokenize(rawsentence)
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return stemmed_tokens


X_train[1]



tfidf_converter = TfidfVectorizer(max_features=1500,min_df=5,max_df=0.7, stop_words=None,tokenizer=stem_text)
X_train_tfidf = tfidf_converter.fit_transform(X_train)
X_test_tfidf = tfidf_converter.transform(X_test) # don't use fit_transform, since we only want to fit on training set, but use the same vectoriser transform on the test

print(tfidf_converter.get_feature_names())

classifier = MultinomialNB().fit(X_train_tfidf, y_train)
y_pred = classifier.predict(X_test_tfidf)
print('confusion matrix:',confusion_matrix(y_test,y_pred))
print('classification report:', classification_report(y_test,y_pred))
print('accuracy:',accuracy_score(y_test, y_pred))

# Write Model Function
def build_model(classifier, X,y, X_test, y_test):
    classifier.fit(X, y)
    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),confusion_matrix(y_test,y_pred),classification_report(y_test,y_pred), y_pred

acc_svm,cm_svm,report_svm,_ = build_model(LinearSVC(), X_train_tfidf,y_train,X_test_tfidf,y_test)
print(acc_svm)
acc_naivebayes,cm_naivebayes,report_naivebayes,_ = build_model(MultinomialNB(), X_train_tfidf,y_train,X_test_tfidf,y_test)
print(acc_naivebayes)
acc_logistic,cm_logistic,report_logistic,_ = build_model(LogisticRegression(penalty='l2', max_iter=100, C=1), X_train_tfidf,y_train,X_test_tfidf,y_test)
print(acc_logistic)


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(cm_svm, index = [i for i in sorted(set(y_test))],
                  columns = [i for i in sorted(set(y_test))])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,cmap="Blues",fmt='g')

print(report_svm)


# A few observations about the prediction results:
# - the model not performing on healthy living, but is getting mostly confused with WELLNESS. Which makes sense, probably these can be grouped together
# - World news can also be confused with politics sometimes
# - on the most part pretty solid performance, but perhaps need more examples for the smaller categories

# ### Next Try n-grams: 1-word and 2-word phrases


import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


tfidf_converter = TfidfVectorizer(min_df=3, max_df=0.9, strip_accents='unicode',tokenizer=tokenize,ngram_range=(1,2),sublinear_tf=1)
X_train_tfidf = tfidf_converter.fit_transform(X_train)
X_test_tfidf = tfidf_converter.transform(X_test) # don't use fit_transform, since we only want to fit on training set, but use the same vectoriser transform on the test


# ### Next Try n-grams: 1-word and 2-word phrases

# Write Model Function
def build_model(classifier, X,y, X_test, y_test):
    classifier.fit(X, y)
    with open('SVCModel.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)
    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred),confusion_matrix(y_test,y_pred),classification_report(y_test,y_pred), y_pred


acc_svm,cm_svm,report_svm,_ = build_model(LinearSVC(), X_train_tfidf,y_train,X_test_tfidf,y_test)
print(acc_svm)


acc_naivebayes,cm_naivebayes,report_naivebayes,_ = build_model(MultinomialNB(), X_train_tfidf,y_train,X_test_tfidf,y_test)
print(acc_naivebayes)
acc_logistic,cm_logistic,report_logistic,_ = build_model(LogisticRegression(penalty='l2', max_iter=100, C=1), X_train_tfidf,y_train,X_test_tfidf,y_test)
print(acc_logistic)


acc_logistic,cm_logistic,report_logistic,_ = build_model(LogisticRegression(penalty='l2',C=4), X_train_tfidf,y_train,X_test_tfidf,y_test)
print(acc_logistic)





