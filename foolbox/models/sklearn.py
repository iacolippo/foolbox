from .base import Model


class SklearnModel(Model):
    """Creates a :class:`Model` instance from a `sklearn` pipeline.

    Parameters
    ----------
    model : `sklearn.pipeline.Pipeline`
        The Sklearn model that should be attacked.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    num_classes : int
        Number of classes for which the model will output predictions.
    preprocessing: 2-element tuple with floats or numpy arrays
        Element-wise preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.

    """
    def __init__(self, model, bounds, num_classes, channel_axis=1, preprocessing=(0, 1)):
        super(SklearnModel, self).__init__(bounds=bounds, channel_axis=channel_axis, preprocessing=preprocessing)
        self._num_classes = num_classes
        self._model = model

    def batch_predictions(self, images):
        images = self._process_input(images)
        images = images.reshape(images.shape[0], -1)
        n = len(images)
        predictions = self._model.predict_proba(images)
        assert predictions.ndim == 2
        assert predictions.shape == (n, self.num_classes())
        assert predictions.sum() - images.shape[0] < 1e-5
        return predictions

    def num_classes(self):
        return self._num_classes
