from classification import NaiveBayes
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils import calculate_accuracy

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = NaiveBayes()
classifier.fit(X_train, y_train)
y_predicted = classifier.predict(X_test)

accuracy = calculate_accuracy(y_predicted, y_test)

print("Accuracy:", accuracy)