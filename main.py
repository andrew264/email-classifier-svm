import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

ham_spam = {'ham': 0, 'spam': 1}

# Read the data
df = pd.read_csv('spam.csv', encoding='utf-8')
df['Label'] = df['Label'].map(ham_spam)

# Remove stopwords from df['EmailText']
nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
df['EmailText'] = df['EmailText'].apply(
    lambda x: ' '.join([ps.stem(word).lower() for word in x.split() if word not in stop_words]))

# Split the data into training and testing sets
train = df.sample(frac=0.75, random_state=69420)
test = df.drop(train.index)
print('Training set size: ', len(train))
print('Testing set size: ', len(test))

# Create a CountVectorizer object and fit it to the data
vectorizer = CountVectorizer()
vectorizer.fit(df['EmailText'])

# Transform the training and testing data using the fitted CountVectorizer
train_data = vectorizer.transform(train['EmailText'])
test_data = vectorizer.transform(test['EmailText'])

# Create an SVM classifier
tuned_parameters = dict(kernel=['rbf', 'linear'], gamma=[1e-3, 1e-4], C=[1, 10, 100, 1000])
model = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, n_jobs=-1)
model.fit(train_data, train['Label'])

# Predict the labels of the test data
pred: np.ndarray = model.predict(test_data)
# Accuracy
print('Model Accuracy: ', round(np.mean(pred == test['Label']) * 100, 3), '%')

# Classification report
print(classification_report(test['Label'], pred))

# Plot the confusion matrix
cm: np.ndarray = confusion_matrix(test['Label'], pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Ham', 'Spam'], rotation=45)
plt.yticks(tick_marks, ['Ham', 'Spam'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# example of a spam email
test_examples = test.sample(n=5)

# Predict the labels of the test emails
inv_ham_spam = {v: k for k, v in ham_spam.items()}
pred: np.ndarray = model.predict(vectorizer.transform(test_examples['EmailText']))
print("Email Text\t\t\tPredicted\tActual")
for i in range(len(test_examples)):
    print(test_examples.iloc[i]['EmailText'][:20],
          '\t', inv_ham_spam[pred[i]],
          '\t', inv_ham_spam[test_examples.iloc[i]['Label']])

# Check against user inputs
while True:
    user_input = input('Enter an email: ')
    if user_input.lower() == 'exit' or user_input == '':
        break
    user_input = ' '.join([ps.stem(word).lower() for word in user_input.split() if word not in stop_words])
    print('Predicted: ', inv_ham_spam[model.predict(vectorizer.transform([user_input]))[0]])
