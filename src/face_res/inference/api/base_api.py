"""
aiar.aioz.truongle - Nov 23, 2021
base api
"""
import abc


class BaseAPI(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _init_modules(self, *args, **kwargs):
        """init all modules to use """
        pass

    @abc.abstractmethod
    def proceed(self, *args, **kwargs):
        """
        main function,
        process input and return output
        """
        pass
