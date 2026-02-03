import time
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=25)

start_time = time.time()
model = LogisticRegression()
model.fit(X, y)
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")