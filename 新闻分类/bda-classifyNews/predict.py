import joblib
import time
import pandas as pd
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

def predict_json(pathToJson):
    df = load_test_set(pathToJson)
    df['short_description'] = df.short_description.apply(process_text)
    df['short_description'] = df.short_description.apply(join_word)
    model, vec = joblib.load('knn_model.joblib')

    feature = vec.transform(df['short_description'])

    prediction = model.predict(feature)

    print('confusion matrix:', confusion_matrix(df['category'], prediction))

    print('classification report:', classification_report(df['category'], prediction))

    print('accuracy_score:', accuracy_score(df['category'], prediction))


def predict(headline, authors, link, des, date):
    model, vec = joblib.load('knn_model.joblib')
    feature = vec.transform([des])
    prediction = model.predict(feature)
    print(prediction[0])


def process_text(text):
    cutwords1 = word_tokenize(text)  # 分词
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义符号列表
    cutwords2 = [word for word in cutwords1 if word not in interpunctuations]  # 去除标点符号
    stops = set(stopwords.words("english"))
    cutwords3 = [word for word in cutwords2 if word not in stops]  # 判断分词在不在停用词列表内
    cutwords4 = []
    for cutword in cutwords3:
        cutwords4.append(PorterStemmer().stem(cutword))  # 词干提取
    return cutwords4


def join_word(text):
    return ' '.join(text)


def load_test_set(path='News_Category.json'):
    data = pd.read_json(path, lines=True)
    return data


link = "https://www.huffpost.com/entry/judge-civil-rights-icon-damon-j-keith-dies_n_5cc63716e4b04eb7ff978fc1"
headline = "Judge And Civil Rights Icon Damon J. Keith Dead At 96"
category = "U.S. NEWS"
short_description = "Keith served more than 50 years in the federal courts."
authors = ""
date = "2019-04-28"

if __name__ == '__main__':
    # 运行：在predict()中填入测试json文件的路径运行即可
    # predict_json('../News_Category.json')

    predict(headline, authors, link, short_description, date)




