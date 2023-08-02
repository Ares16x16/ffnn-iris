from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import ffnn
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
lb = LabelBinarizer()
y = lb.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ffnn.FFNN(input_size=X.shape[1], hidden_size=5, output_size=y.shape[1])
model.train(X_train, y_train, epochs=1000, learning_rate=0.01)

y_pred = np.argmax(model.forward(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2%}")