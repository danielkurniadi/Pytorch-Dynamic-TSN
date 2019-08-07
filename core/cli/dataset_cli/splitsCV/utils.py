import os
import glob

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from core.dataset.utils import search_files_recursively


#------------------------
# Helpers
#------------------------

def get_path_label_list(
    dataset_dir
):
    """List paths and labels of all data in a directory.
    This assumes that all data are located at 2 levels below the directory (after class_folder)
    """
    all_paths, all_labels = [], []      # place holder for all data in directory

    class_folders = filter(
        os.path.isdir,
        glob.glob(dataset_dir + '/*')   # get class folders' paths
    )

    for label, class_folder in enumerate(class_folders):    # label encode using for-loop index
        data_paths = glob.glob(class_folder + '/*')         # get subfolders in each class folder
        data_labels = [label] * len(data_paths)
        
        all_paths.extend(data_paths)        # add those subfolders to list
        all_labels.extend(data_labels)      # add labels to list

    return sorted(all_paths), sorted(all_labels)


def train_test_split(
    all_paths,
    all_labels,
    test_size = 0.1,
    random_state = 42
):
    all_paths = np.array(all_paths)     # convert to numpy to ease multi-indexing
    all_labels = np.array(all_labels)   # convert to numpy to ease multi-indexing

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(iter(sss.split(all_paths, all_labels)))

    return (
        all_paths[train_idx], all_paths[test_idx],
        all_labels[train_idx], all_labels[test_idx]
    )


def write_metadata_to_split_file(
    outfile,
    data_paths,
    data_labels,
    split_type = 'II',
    data_prefix = ''
):
    """Write all metadata to a file
    """
    if split_type in ['I', 'i', 1]:
        # data is represented by a file
        to_tmpl = "{} {}\n"
        to_writes = zip(data_paths, data_labels)

    elif split_type in ['II', 'ii', 2]:
        # data is represented by a (sub)folder
        num_frames_list = []
        for data_path in data_paths:
            abs_path = os.path.abspath(data_path)
            files = search_files_recursively(abs_path, prefix=data_prefix)

            if not files:
                # TODO: use warn.warnings
                print(
                    ".. Folder split has num frames == zero for %s; " 
                    "Skipping..." % abs_path
                )
                continue
            num_frames = len(files)
            num_frames_list.append(num_frames)

        to_tmpl = "{} {} {}\n"
        to_writes = zip(data_paths, num_frames_list, data_labels)
    
    with open(outfile, 'w+') as fp:
        # write or create mode
        for to_write in to_writes:
            fp.write(to_tmpl.format(*to_write))


#------------------------
# Main utilities
#------------------------

def generate_skf_split_files(
    all_paths,
    all_labels,
    outdir,
    include_test_split = True,
    split_type = 'file',
    split_prefix = "",
    n_splits = 5,
    data_prefix = '',
    random_seed = 42
):
    """
    Generaterate Shuffled-Stratified K Fold split 
    given paths and label to dataset.
    """
    if include_test_split:
        # if True, split data to train, val, test. 
        # otherwise just train and val split
        test_splitname = '{}_test_split.txt'.format(split_prefix)
        test_splitpath = os.path.join(outdir, test_splitname)

        all_train_paths, all_test_paths, all_train_labels, all_test_labels =  train_test_split(
            all_paths, all_labels, test_size = 0.1, random_state = random_seed
        )
        # writing metadata for test split
        write_metadata_to_split_file(
            test_splitpath,
            all_test_paths,
            all_test_labels,
            split_type,
            data_prefix
        )

    else:
        # here, we consider train and val as part of training 
        # (a.k.a development) phase, hence the naming below for paths & label
        all_train_paths, all_train_labels = all_paths, all_labels

    # stratify dataset on train and validation
    skf = StratifiedShuffleSplit(
        n_splits = n_splits,
        test_size = 0.2,
        random_state = random_seed
    )

    for i, (train_idx, val_idx) in enumerate(skf.split(all_train_paths, all_train_labels)):
        # train split
        X_train = all_train_paths[train_idx]    # X_train is list of train data path 
        y_train = all_train_labels[train_idx]   # y_train is list of label values

        train_splitname = "{}_train_split_{}.txt".format(split_prefix, i)
        train_splitpath = os.path.join(outdir, train_splitname) 

        # writing metadata for training split
        write_metadata_to_split_file(
            train_splitpath,
            X_train,
            y_train,
            split_type,
            data_prefix
        )

        # validation split
        X_val = all_train_paths[val_idx]        # X_val is list of val data path
        y_val = all_train_labels[val_idx]       # y_val is list of val data path

        val_splitname = "{}_val_split_{}.txt".format(split_prefix, i)
        val_splitpath = os.path.join(outdir, val_splitname) 

        # writing metadata for validation split
        write_metadata_to_split_file(
            val_splitpath,
            X_val,
            y_val,
            split_type,
            data_prefix
        )
