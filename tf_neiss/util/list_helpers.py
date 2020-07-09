import logging
import os
from collections import Iterable, Iterator
from numbers import Number

from util.io_helpers import DefaultIOContext


class ListMixDefinition(object):
    def __init__(self, list_filenames, mixing_ratio, io_context=DefaultIOContext()):
        """
        JUST a container holding some information and checking if images exist
        """
        assert len(list_filenames) == len(mixing_ratio)
        assert len(list_filenames) > 0
        assert isinstance(list_filenames, list)
        assert isinstance(mixing_ratio, list)
        for list_filename in list_filenames:
            assert isinstance(list_filename, str)
            assert os.path.isfile(io_context.get_io_filepath(list_filename)), \
                "{} does not exist!".format(io_context.get_io_filepath(list_filename))
        for part in mixing_ratio:
            assert isinstance(part, Number)
        self._list_filenames = list_filenames
        self._mixing_ratio = mixing_ratio

    @property
    def list_filenames(self):
        """

        :rtype: list[str]
        """
        return self._list_filenames

    @property
    def mixing_ratio(self):
        """

        :rtype: list[Number]
        """
        return self._mixing_ratio


class FileListProviderFn(object):
    def __init__(self, file_name):
        self._file_name = file_name
        self._logger = logging.getLogger(str(type(self)))
        self._file_list = None

    def _load_list(self):
        retval = []
        with open(self._file_name, 'r') as list_file:
            for img_fn in [line.rstrip('\n ') for line in list_file]:
                if img_fn is not None and len(img_fn) > 0:
                    retval.append(img_fn)
        return retval

    def get_list(self):
        """
        :param io_context:
        :type io_context: IOContext
        :param cache:
        :type cache: dict
        :return:
        :rtype: list of tuples (filename, reference/transcript, length/width)
        """
        if self._file_name is None:
            return None
        if self._file_list is None:
            self._file_list = self._load_list()
        else:
            self._logger.info("reusing: " + self._file_name)
        return self._file_list


class FileListIterablor(Iterator, Iterable):
    def __init__(self, file_list_provider):
        """
        :param file_list_provider:
        :type file_list_provider: FileListProvider
        """
        self._file_list = file_list_provider.get_list()
        self._index = -1

    def shuffle(self, rng):
        """

        :param rng: random number generator
        :type rng: Random
        """
        rng.shuffle(self._file_list)

    def __iter__(self):
        return self

    def next(self):
        self._index = (self._index + 1) % len(self._file_list)
        return self._file_list[self._index]

    def __next__(self):
        return self.next()
