import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

# Read the data
df = pd.read_csv('spam.csv', encoding='utf-8')

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
pred: np.ndarray = model.predict(vectorizer.transform(test_examples['EmailText']))
print("EmilText\t\t\tPredicted\tActual")
for i in range(len(test_examples)):
    print(test_examples.iloc[i]['EmailText'][:20], '\t', pred[i], '\t', test_examples.iloc[i]['Label'])
