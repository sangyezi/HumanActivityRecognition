import timeit


class ReduceDimensionBaseClass:
    def __init__(self, **kwargs):
        self.learner = None
        self.trained = False

    def reduceDimension(self, xdata):
        if self.trained:
            start  = timeit.default_timer()
            xdata_transformed = self.learner.transform(xdata)
            stop = timeit.default_timer()
            return [stop - start, xdata_transformed]
        else:
            raise RuntimeError("Error: cannot transform, not trained yet")
