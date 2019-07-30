import os
from pathlib import Path
from multiprocessing import Pool, current_process


#-----------------------
# Write
#-----------------------

def safe_mkdir(directory):
    try: 
        os.mkdir(directory)
    except (FileExistsError, OSError):
        pass  # folder has been created previously

    return os.path.abspath(directory)


#-----------------------
# Search
#-----------------------

def abs_listdir(
    directory
):
    """
    Parameters:
        .. directory (str): abspath to directory/folder
    
    Returns:
        .. files (list): abspaths to files in directory
    """
    return [
        os.path.abspath(os.path.join(directory, f))
        for f in os.listdir()
    ]


def search_files_recursively(
    directory,
    prefix,
    by_extensions = [],
    abspaths = True
):
    """
    Parameters:
        .. directory (str): abspath to directory
        .. by_extensions (list/array): extension file to look for. None means look for any file.
    
    Returns:
        .. files (list): all files that matches with extension (and other filter if added in the future...)
    """
    if by_extensions:
        patterns = list(map(lambda x: '**/' + prefix + '*' + x, by_extensions))
        files = []
        for pattern in patterns:
            files.extend(Path(directory).glob(pattern))

    else:
        files = list(Path(directory).glob('**/' + prefix + '*.*'))
    
    if abspaths:
        return sorted([
            os.path.abspath(str(f))
            for f in files
        ])

    else:
        return sorted([
            str(f) for f in files
        ])


#-----------------------
# Others
#-----------------------

def clean_filename(filename):
    """
    """
    return filename.replace(' ', '_').replace('.','')


def get_basename(filename):
    """
    Parameters:
        .. filename (str): file name with extension

    Returns:
        .. basename (str): file name without extension
    """
    basename, _ = os.path.splitext(os.path.basename(filename))
    return basename
