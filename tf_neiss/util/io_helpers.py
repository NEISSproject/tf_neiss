import errno
import os
import shutil


def cp_copy(src, dest, exist_ok=True):
    """create path copy"""
    if not os.path.isdir(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if not exist_ok:
        if os.path.isfile(dest) or os.path.isdir(dest):
            raise IOError("File already exist! {}".format(dest))
    shutil.copy(src, dest)


def file_path_with_mkdirs(path_with_filen_name):
    """will create all dirs to save the file to the given path"""
    if not os.path.isdir(os.path.dirname(path_with_filen_name)):
        try:
            os.makedirs(os.path.dirname(path_with_filen_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return path_with_filen_name


class IOContext(object):
    def get_io_filepath(self, filepath):
        """

        :param filepath:
        :type filepath: unicode
        """
        raise NotImplementedError()


class DefaultIOContext(IOContext):
    def get_io_filepath(self, filepath):
        return filepath


class ReplaceIOContext(IOContext):
    def __init__(self, old, new):
        """

        :param old:
        :type old: unicode
        :param new:
        :type new: unicode
        """
        self._old = old
        self._new = new

    def get_io_filepath(self, filepath):
        return filepath.replace(self._old, self._new, 1)
