"""Module for IMDB data fetching and loading 
"""

import requests
import os
import logging
import shutil


logger = logging.getLogger(__name__)


def download_imdb_data():
    """Download imdb raw data.
    
    The data will be saved into the .imdb_data folder under current working directory. If .imdb_data
    folder already exists, no download occurs.
    
    Returns: 
        tuple:
            bool: whether download occurs
            string: folder path

    """
    data_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    data_folder = '.imdb_data'
    download_file_name = 'aclImdb_v1.tar.gz'
    
    if os.path.isdir(data_folder):
        logger.info(f'Folder {data_folder} already exits')
        return False, data_folder
    else:
        logger.info(f'Create folder {data_folder}')
        os.mkdir(data_folder)
        
        logger.info(f'Download data from {data_url}')
        download_path = os.path.join(data_folder, download_file_name)
        data_req = requests.get(data_url, stream=True)
        if data_req.status_code == 200:
            with open(download_path, 'wb') as outfile:
                # reduce the memory use
                for chunk in data_req.iter_content(chunk_size=128):
                    outfile.write(chunk)
        else:
            raise Exception(f'Fail to download file from {data_url}')
        
        logger.info(f'Extract downloaded data from {download_path}')
        shutil.unpack_archive(download_path, '.imdb_data')
        
        return True, data_folder
    

def get_imdb_raw():
    """Gets IMDB raw data.
    
    Returns:
        tuple:
            list: feature data for training
            list: label data for training
            list: feature data for testing
            list: label data for testing
    
    """
    downloaded, foldername = download_imdb_data()
    
    if downloaded:
        logger.info(f'Data Downloaded into folder {foldername}')
    else:
        logger.info((f'Target data folder {foldername} already exisits.'
                     'If function not properly works, remove that folder and rerun to download data.'))

    path_dict = {}
    path_dict['train_pos'] = os.path.join(foldername, 'aclImdb', 'train', 'pos')
    path_dict['train_neg'] = os.path.join(foldername, 'aclImdb', 'train', 'neg')
    path_dict['test_pos'] = os.path.join(foldername, 'aclImdb', 'test', 'pos')
    path_dict['test_neg'] = os.path.join(foldername, 'aclImdb', 'test', 'neg')

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for key, path in path_dict.items():
        rev_list = []
        filenames = os.listdir(path)

        filenames = [_ for _ in filenames if _.endswith('.txt')]

        for curr_filename in filenames:
            curr_filepath = os.path.join(path, curr_filename)

            with open(curr_filepath, 'r') as infile:
                curr_rev = infile.read()
                rev_list.append(curr_rev)
                
        if key.startswith('train'):
            curr_data = train_data
            curr_label = train_label
        else:
            curr_data = test_data
            curr_label = test_label
            
            
        if key.endswith('pos'):
            label_val = 1
        else:
            label_val = 0
            
        curr_data.extend(rev_list)
        curr_label.extend([label_val] * len(rev_list))
        
    return train_data, train_label, test_data, test_label
