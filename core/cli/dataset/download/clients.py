import os
import requests
import warnings

import kaggle


def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename


class URLClient(object):
    """
    Client with any url for file/folder download. 
    Assuming no session/authentication required to download
    """

    def __init__(
        self,
        url,
        save_folder,
        unzip
    ):
        self.url = url
        self.save_folder = save_folder
        self.unzip = unzip

    def __call__(self):
        file_url = self.url
        self.downloaded_file_path = download_file(file_url)


class DataAPIClient(object):
    """
    Data API Client abstract class
    """

    __abstract__ = True

    def __init__(
        self,
        dataset,
        save_folder,
        credential_file,
        unzip = True
    ):
        self.dataset = dataset
        self.save_folder = save_folder
        self.credential_file = credential_file
        self.unzip = unzip


class Kaggle(DataAPIClient):
    """
    Kaggle dataset API
    """
    KAGGLE_JSON_PATH = os.path.join(os.getenv('HOME'), '.kaggle/kaggle.json')

    def __init__(
        self,
        dataset,
        save_folder,
        credential_file = None),
        unzip = True
    ):  
        if credential_file:
            warnings.warn("- Put your kaggle.json at %s instead." % self.KAGGLE_JSON_PATH)

        super(self, Kaggle).__init__(
            dataset, save_folder, KAGGLE_JSON_PATH, unzip
        )

    def __call__(self):
        if not os.path.isfile(self.credential_file):
            # check if kaggle file exists
            warnings.warn("- Kaggle Credential File (kaggle.json) not found." 
                "Using Environment variables for Kaggle USERNAME and KEY.")
            
            username = os.environ.get('KAGGLE_USERNAME')
            key = os.environ.get('KAGGLE_KEY')
            
            if (not username) and (not key):
                # check if kaggle env var exists as subtitute
                raise OSError("Couldn't find kaggle.json nor environment variables for"
                    "USERNAME and KEY. Please do export those variables or create kaggle.json file."
                    "See Kaggle API docs")

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            self.dataset,
            path=self.save_folder,
            unzip=True
        )
