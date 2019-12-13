import glob
import logging
import os
import shutil
import sys

"""script to divide a folder with generated/training data into a train and val folder
    - val folder contains 500 Samples if not changed in source code
    - DOES NOT work if images structured in subfolders, see below
    - if there is no dir in the given folder -> split this folder
    - if there are dir/s in the folder -> perform split on each folder
    - split on sorted list -> repeated runs should give the same result
    """


def main(args):
    foldername = args[1]
    print("CWD: {}".format(os.getcwd()))
    print("foldername: {}".format(foldername))

    dirs = os.walk(foldername).next()[1]
    dirs = [os.path.join(foldername, x) for x in dirs]
    print(dirs)
    if len(dirs) == 0:
        print("no subdirs found -> run directly on {}".format(foldername))
        dirs = [foldername]

    for dir in dirs:
        print("perform split on {}".format(dir))
        dir_path = dir
        # image_list = sorted(glob.glob1(os.path.join(foldername, dir_path), "*.jpg"))
        image_list = sorted(glob.glob1(dir_path, "*.jpg"))
        # image_list = sorted(glob.glob1(dir_path , "*.png"))
        if len(image_list) == 0:
            logging.error("Could not find any '*.jpg' in {}".format(dir_path))
            exit(1)
        else:
            print("  found {} images".format(len(image_list)))

        # val_len = int(len(image_list) * 0.1)
        val_len = int(500)

        val_list = image_list[:val_len]
        train_list = image_list[val_len:]
        #  save first 10%/500 of list to val list

        for subdir, part_list in zip(["val", "train"], [val_list, train_list]):
            os.makedirs(os.path.join(dir_path, subdir))
            print("  move files in {}...".format(subdir))
            for image_file in part_list:
                shutil.move(os.path.join(dir_path, image_file), os.path.join(dir_path, subdir, image_file))
                try:
                    shutil.move(os.path.join(dir_path, image_file + ".txt"),
                                os.path.join(dir_path, subdir, image_file + ".txt"))
                except IOError as ex:
                    print(ex)
                try:
                    shutil.move(os.path.join(dir_path, image_file + ".info"),
                                os.path.join(dir_path, subdir, image_file + ".info"))
                except IOError as ex:
                    pass

            print("  write list: {}...".format(os.path.join(dir_path, "{}_{}.lst".format(dir_path, subdir))))
            with open(os.path.join(foldername, "{}_{}.lst".format(os.path.basename(dir_path), subdir)), "w") as fobj:
                fobj.writelines([os.path.join(dir_path, subdir, x) + "\n" for x in part_list])


if __name__ == '__main__':
    main(sys.argv)
