import os
import glob

from sklearn.model_selection import StratifiedShuffleSplit


#------------------------
# Helpers
#------------------------

def train_test_split(
    all_paths,
    all_labels,
    test_size = 0.1,
    random_state = 42
):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_idx, test_idx = next(iter(sss.split(all_paths, all_labels)))

    return (
        all_paths[train_idx], all_paths[test_idx],
        all_labels[train_idx], all_labels[test_idx]
    )


def append_metadata_to_split_file(
    outfile,
    data_paths,
    data_labels,
    split_type = 'file',
):
    if split_type == 'file':
        to_tmpl = "{} {}\n"
        to_writes = zip(data_paths, data_labels)

    elif split_type == 'folder':
        data_num_frames = []
        
        for data_path in data_paths
            num_frames = len([
                file for file in os.listdir()
                if os.path.isfile(os.path.join(data_path, file))
            ])
        
            if num_frames == 0:
                print("Danger! Folder split has num frames == zero for %s" % data_path)
            data_num_frames.append(num_frames)

        to_tmpl = "{} {} {}\n"
        to_writes = zip(data_paths, data_num_frames, data_labels)
    
    with open(outfile, 'w+') as fp:
        for metadata in to_writes:
            fp.write(to_tmpl.format(*metadata))


#------------------------
# Main utilities
#------------------------

def generate_skf_split_files(
    all_paths,
    all_labels,
    outdir,
    include_test_split = True,
    split_type = 'file',
    out_prefix = "",
    n_splits = 5
):
    """Generaterate Shuffled-Stratified K Fold split given paths and label to dataset.
    """
    if include_test_split:
        test_splitname = '{}_test_split.txt'.format(out_prefix)
        test_splitpath = os.path.join(outdir, test_splitname)

        all_train_paths, all_test_paths, all_train_labels, all_test_labels =  train_test_split(
            all_paths, all_labels, test_size = 0.1, random_state = 42
        )
        
        append_metadata_to_split_file(
            test_splitpath,
            all_test_paths,
            all_test_labels,
            split_type
        )

    else:
        all_train_paths, all_train_labels = all_paths, all_labels

    # stratify dataset on train and validation
    skf = StratifiedShuffleSplit(
        n_splits = n_splits,
        test_size=0.2
    )

    for i, (train_idx, val_idx) in enumerate(skf.split(all_train_paths, all_train_labels)):
        # train split
        X_train = all_train_paths[train_idx]    # X_train is list of train data path 
        y_train = all_train_labels[train_idx]   # y_train is list of label values

        train_splitname = "{}_train_split_{}.txt".format(out_prefix, i)
        train_splitpath = os.path.join(outdir, train_splitname) 

        append_metadata_to_split_file(
            train_splitpath,
            X_train,
            y_train,
            split_type
        )

        # validation split
        X_val = all_train_paths[val_idx]        # X_val is list of val data path
        y_val = all_train_labels[val_idx]       # y_val is list of val data path

        val_splitname = "{}_val_split_{}.txt".format(out_prefix, i)
        val_splitpath = os.path.join(outdir, val_splitname) 

        append_metadata_to_split_file(
            val_splitpath,
            X_val,
            y_val,
            split_type
        )
