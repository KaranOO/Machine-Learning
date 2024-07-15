import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        # calcuate mean var and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._prior = np.zeros(n_classes, dtype = np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis = 0)
            self._var[idx, :] = X_c.var(axis = 0)
            self._prior[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        #calculate posteriors prob for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        #return class with the highest pred
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx,x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 ** var))
        deno = np.sqrt(2 * np.pi * var)
        return numerator / deno

#Testing
if __name__ == "__main__":
    #inports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets as df

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = df.make_classification(n_samples = 1000, n_features = 10, n_classes = 2, random_state = 123)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    nb = NaiveBayes()

    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes Classification Accuracy: ", accuracy(y_test, predictions))

# X is the input & y is the output