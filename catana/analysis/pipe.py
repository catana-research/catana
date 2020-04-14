"""
Pipeline
--------

"""
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import _BaseComposition


class StepObject(_BaseComposition):

    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)


class Pipe(object):

    def __init__(self, *args, **kwargs):
        self.pipeline = Pipeline(*args, **kwargs)

    def _add_step(self, name, step):
        self.pipeline.steps.append(name, step)
