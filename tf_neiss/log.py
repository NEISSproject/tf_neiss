import glob
import io
import logging
import os
import sys


class Tee(io.FileIO):
    def __init__(self, file_path, console=True, delete_existing=True, *args, **kwargs):
        """redirect to all console out put to file_path, redirect stderr to stdout!,
        -use self.close() to set previous behaviour"""
        self._file_path = file_path
        self._console = console
        self.stdout_origin = sys.stdout
        self.stderr_origin = sys.stderr

        if delete_existing:
            mode = "w"
        else:
            mode = "a"
        super(Tee, self).__init__(file_path, mode=mode, *args, **kwargs)

        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def write(self, data):
        if self._console:
            self.stdout_origin.write(data)
        with open(self._file_path, 'a') as fobj:
            fobj.write(data)

    def close(self):
        sys.stdout = self.stdout_origin
        sys.stderr = self.stderr_origin


def get_commit_id(repo_dir=None, recursive=True):
    """
    returns the git commit id from given "repo_dir"
    -walks upwards if given path does not contain a git repo
    -starts with this file location if no repo_dir is given
    returns empty string if no git-repo can be found
    :return commit_id, actual_repo_path"""

    def get_head_path(in_dir):  # recursive function
        try:
            with open(os.path.join(in_dir, ".git/HEAD"), 'r') as f_obj:
                head_path_ = f_obj.readline()[5:-1]
        except IOError:
            if os.path.dirname(in_dir) != in_dir and recursive:
                head_path_, in_dir = get_head_path(os.path.dirname(in_dir))
            else:
                logging.error("in get_commit_id(): can not find git-repo from: " + os.path.join(repo_dir))
                return "", repo_dir
        return head_path_, in_dir

    if repo_dir is None:
        repo_dir = os.path.realpath(__file__)
    elif repo_dir[0] != "/":  # make path absolute
        repo_dir = os.path.join(os.getcwd(), repo_dir)

    head_path, actual_repo = get_head_path(repo_dir)
    if head_path == "":
        return "", repo_dir

    try:
        with open(os.path.join(actual_repo, ".git/" + str(head_path)), 'r') as file_obj:
            commit_id = file_obj.readline()[:-1]
    except IOError:
        logging.error("Error in get_commit_id(): invalid path in 'HEAD'" + os.path.join(actual_repo))
        return "", actual_repo

    return commit_id, actual_repo


def get_path_from_exportdir(model_dir, pattern, not_pattern='cm_tc.txt'):
    export_dir = os.path.join(model_dir, "export")
    print(os.path.abspath(export_dir))
    print(glob.glob(export_dir))
    name = [x for x in glob.glob1(export_dir, pattern) if not_pattern not in x]
    if len(name) == 1:
        return os.path.join(export_dir, name[0])
    else:
        raise IOError(
            "Found {} '{}' files in {}, there must be exact one. Use flag to specify otherwise".format(len(name),
                                                                                                       pattern,
                                                                                                       export_dir))


if __name__ == "__main__":
    print("test print redirection")
    print("Hello only Console")
    tee = Tee("test_log.txt", )
    print("print to both")
    tee.close()
    print("as before")
    Tee("test_log.txt", console=False, delete_existing=False)
    print("only to file")
    del tee
    print("as before after only to file")
