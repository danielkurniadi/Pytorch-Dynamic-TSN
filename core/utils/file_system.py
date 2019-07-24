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

def abs_listdir(directory):
    """
    Parameters:
        .. directory (string)
    
    Returns:
        .. files (list): list of file path (abspath) in directory
    """
    return [
        os.path.join(directory, filename)
        for filename in os.listdir() 
    ]


def search_files_recursively(
    directory,
    by_extensions = None,
    abspaths = True
):
    """
    Parameters:
        .. directory (string): abspath to directory
        .. by_extensions (list/array): extension file to look for. None means look for any file.
    
    Returns:
        .. files (list): all files that matches with extension (and other filter if added in the future...)
    """
    if by_extensions:
        patterns = list(map(lambda x: '**/*'+x, by_extensions))
        files = []
        for pattern in patterns:
            files.extend(Path(video_dir).glob(pattern))

    else:
        files = os.listdir(directory)
    
    if abspaths:
        files = [
            os.path.join(directory, f)
            for f in files
        ]

    return files


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