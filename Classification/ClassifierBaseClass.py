import timeit


class ClassifierBaseClass:

    def __init__(self, **kwargs):
        self.trained = False

    def train(self, xtrain, ytrain):
        self.setClassifer()

        start_train = timeit.default_timer()
        self.classifier.fit(xtrain, ytrain)
        stop_train = timeit.default_timer()

        self.trained = True
        return (stop_train - start_train)

    def setClassifer(self):
        self.classifier = None



    def test(self, xtest, ytest):
        if self.trained:
            start_test = timeit.default_timer()
            accuracy = self.classifier.score(xtest, ytest)
            stop_test = timeit.default_timer()
            return [stop_test - start_test, accuracy]
        else:
            raise RuntimeError("Error: cannot test, classifer not trained")