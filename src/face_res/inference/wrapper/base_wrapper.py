"""
aiar.aioz.truongle - Nov 23, 2021
base wrapper
"""
import abc


class BaseWrapper(object, metaclass=abc.ABCMeta):
    def __init__(self):
        self._model = None  # this is variable store model
        self._device = None  # ste device

    @abc.abstractmethod
    def init(self):
        """load model anf init (if any)"""

    @abc.abstractmethod
    def _load_model(self, *args, **kwargs):
        """load model from weight"""
        pass

    def _check_model_init(self, logger=None):
        if self._model is None:
            if logger:
                logger.warning("model has not been initialized. Re-initialize model now ...")
            self.init()

    @abc.abstractmethod
    def _pre_process(self, *args, **kwargs):
        """
        pre-process input data before processing
        """
        pass

    @abc.abstractmethod
    def _post_process(self, *args, **kwargs):
        """
        post-process output
        """
        pass

    @abc.abstractmethod
    def process_prediction(self, *args, **kwargs):
        """
        main function,
        process input and return output
        """
        pass
