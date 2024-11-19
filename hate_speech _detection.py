from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
from sklearn import metrics
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string


nltk.download('stopwords')

# Load the stopwords
stop_words = set(stopwords.words('english'))

data = pd.read_csv("labeled_data.csv")
#print(data.head())
data["labels"] = data["class"].map({0: "Hate Speech",1: "Offensive Language",2: "No Hate and Offensive"})
#print(data.head())


data = data[["tweet", "labels"]]
#print(data.head())


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*','', text)
    # Use the loaded stopwords set here
    text = [word for word in text.split(' ') if word not in stop_words]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean)


x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


sample="SouthCarolina is full of white trash"

data=cv.transform([sample]).toarray()


print(clf.predict(data))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))