#Name: George Eralil
#ID: 11588978
import numpy as np
import pandas as pd
import os
import itertools
import nltk
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

#function to generate confusion matrix plot for evaluation
#code for plot_confusion_matrix courtesy: https://www.datacamp.com/community/tutorials/scikit-learn-fake-news
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#start of application
# creating location varible for real-time path finding #
_location_ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

# reading the data
df = pd.read_csv(os.path.join(_location_,'FakeAndRealSample.csv'))
df.head(10)

#filling nan values with space
df = df.fillna(' ')

#preprocessing the data by removing stop words and tokenizing
#and lemmatizing the sentences
stop_words = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

for index, row in df.iterrows():
    tokenized_sentence = ' '
    sentence = row['title']
    #removing whitespaces and special characters in sentences
    sentence = re.sub(r'[^\w\s]', '', sentence)
    #word tokenization
    words = word_tokenize(sentence)
    #stopwords removal
    words = [w for w in words if not w in stop_words]
    #lemmatization
    for words in words:
        tokenized_sentence = tokenized_sentence + ' ' + str(lemmatizer.lemmatize(words)).lower()

        df.loc[index, 'title'] = tokenized_sentence

df.head(10)

df.label = df.label.astype(str)
df.label = df.label.str.strip()

#changing true labels to 1 and fake labels to 0
dict = {'TRUE' : '1' , 'FAKE' : '0'}

df['label'] = df['label'].map(dict)
df.head(10)
df = df.dropna()

#vectorizing the titles using tfidf
x_df = df['title']
y_df = df['label']

#Splitting data in test and train data
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size = 0.2, random_state = 7)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df = 0.7, norm = "l2")

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

print(tfidf_train)
print(tfidf_test)

#Initializing classifier.
#Run each code section for classifier separately.

#Initializing Passive Aggressive Classifier
passive_classifier = PassiveAggressiveClassifier(max_iter=50)
passive_classifier.fit(tfidf_train, y_train)

y_pred = passive_classifier.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy of Passive aggressive classfier: {round(score*100,2)}%')
cm = confusion_matrix(y_test, y_pred, labels=['0', '1'])
plot_confusion_matrix(cm, classes=['Fake', 'Real'])

#Initializing Multinomial Naive Bayes Classifier
multi_Bayes = MultinomialNB()

multi_Bayes.fit(tfidf_train, y_train)
y_pred = multi_Bayes.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy of Multinomial Naive Bayes: {round(score*100,2)}%')
cm = confusion_matrix(y_test, y_pred, labels=['0', '1'])
plot_confusion_matrix(cm, classes=['Fake', 'Real'])

#Initializing Logistic Regression Classifier
logistic_reg = LogisticRegression()
logistic_reg.fit(tfidf_train, y_train)
y_pred = logistic_reg.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f'Accuracy of Logistic Regression: {round(score*100,2)}%')
cm = confusion_matrix(y_test, y_pred, labels=['0', '1'])
plot_confusion_matrix(cm, classes=['Fake', 'Real'])

#Initializing Decision Trees
k_decision_tree = DecisionTreeClassifier()
k_decision_tree.fit(tfidf_train, y_train)
y_pred = k_decision_tree.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f'Accuracy of Decision Trees: {round(score*100,2)}%')
cm = confusion_matrix(y_test, y_pred, labels=['0', '1'])
plot_confusion_matrix(cm, classes=['Fake', 'Real'])

#Initializing Perceptron algorithm
perceptron = Perceptron()
perceptron.fit(tfidf_train, y_train)
y_pred = perceptron.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f'Accuracy of Perceptron: {round(score*100,2)}%')
cm = confusion_matrix(y_test, y_pred, labels=['0', '1'])
plot_confusion_matrix(cm, classes=['Fake', 'Real'])