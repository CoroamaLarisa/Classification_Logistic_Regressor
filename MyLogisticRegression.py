from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


class MyLogisticRegression:
    def __init__(self):
        self.intercept_ = 0
        self.coef_ = []

    def fit(self, x, y, learningRate=0.1, noEpochs=1000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]

        for epochs in range(noEpochs):

            for i in range(len(x)):
                y_computed = sigmoid(self.eval(x[i], self.coef_))

                error = y_computed - y[i]

                for j in range(len(x[0])):
                    self.coef_[j + 1] = self.coef_[j + 1] - learningRate * error * x[i][j]

                self.coef_[0] = self.coef_[0] - learningRate * error

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def eval(self, xi, coef_):
        
        yi = coef_[0]

        for j in range(len(xi)):
            yi += coef_[j + 1] * xi[j]

        return yi

    def predictOneSample(self, sample):
        threshold = 0.5
        computedOutputs = sigmoid(self.eval(sample, [self.intercept_] + self.coef_))
        if computedOutputs <= threshold:
            return 0
        else:
            return 1

    def predict(self, x):
        return [self.predictOneSample(xi) for xi in x]
