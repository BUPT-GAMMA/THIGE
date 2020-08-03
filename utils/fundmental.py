import os


def create_buckets(base_path, suffices, delim='_', remove_path=0):

    path = base_path + delim.join([str(x) if type(x) is int or type(x) is float else x for x in suffices]) + "/"
    # path = base_path + delim.join(suffices) + "/"
    if remove_path == 1:
        os.system("rm -rf {}".format(path))
    if not os.path.exists(path):
        os.makedirs(path)
        print('Path\t{}\t has been created!\n'.format(path))
    else:
        print('Path\t{}\t are not empty!\n'.format(path))
    return path
