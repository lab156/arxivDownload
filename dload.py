import os
import subprocess
import hashlib
import xml.etree.ElementTree as ET
import pandas as pd
import sys
import logging


def parse_element(elem):
    return_dict = {}
    for e in elem:
        return_dict[e.tag] = e.text # loop element and extract info
    return return_dict
def parse_root(root):
        return [parse_element(child) for child in iter(root) \
                if child.tag != 'timestamp']

def parse_manifest(xml_path):
    '''
    xml_path: the path of the xml manifest file
    output: a pandas dataframe of all the files in the src bucket

    ### If you want to create an allfiles.csv ###
    1) Download the manifest from aws: $s3cmd get --requester-pays s3://arxiv/src/arXiv_src_manifest.xml
    2) On the jupyter notebook: Parsing arxiv manifest and querying metadata do
    new_all = parse_manifest(FilePath.xml)
    new_all.to_csv(FilePath.csv)
    '''
    with open(xml_path, 'r') as f:
        mani = ET.parse(f)
    root = mani.getroot()
    return pd.DataFrame(parse_root(root))


class DownloadMan(object):
    def __init__(self,
            mountpoint,
            allfiles, 
            downloaded_log,
            error_log,
            s3_url = 's3://arxiv/'):
        '''
        mountpoint is root file to download
        allfiles: csv file with the all the files 
        downloaded_log is a csv file list of previously downloaded files 
        error_log: text files with the error message
        '''
        self.allfiles_path = os.path.join(mountpoint, allfiles)
        self.allfiles_df = pd.read_csv(os.path.join(mountpoint, allfiles), 
                index_col=0)
        self.downloaded_path = os.path.join(mountpoint, downloaded_log)
        self.downloaded_df = pd.read_csv(self.downloaded_path, index_col=0)
        self.error_log_path = os.path.join(mountpoint, error_log)
        self.mountpoint = mountpoint
        self.s3_url = s3_url



    def next_file(self):
        '''
        gets the next file that has not been downloaded
        return a Pandas series
        filename string is an attribute
        '''
        idx = self.allfiles_df.filename.isin(self.downloaded_df.filename)
        try:
            return self.allfiles_df[~idx].iloc[0]
        except IndexError as e:
            logging.info("No next_file found")
            raise e

    def get(self, filename):
        file_save_path = os.path.join(self.mountpoint, filename)
        file_s3_path = self.s3_url + filename
        P = subprocess.run(
                ['/usr/bin/s3cmd', 'get', '--requester-pays', 
                    file_s3_path, file_save_path],
                   stderr=subprocess.PIPE,
                   stdout=subprocess.PIPE)
        if P.returncode:
            print('Error Downloading file: %s'%filename)
            with open(self.error_log_path, 'a') as efile:
                efile.write(str(P.stderr) + '\n')
        else:
            print('Dowloand of file %s successful'%filename)
            append_df = pd.DataFrame([{'filename': filename}])
            self.downloaded_df = self.downloaded_df.append(append_df,
                    ignore_index=True)
            self.downloaded_df.to_csv(self.downloaded_path)
        return P.returncode


    def get_next(self):
        return self.get(self.next_file().filename)

    def md5sum(self, filename):
        '''
        compute the md5 sum of filename
        '''
        file_save_path = os.path.join(self.mountpoint, filename)
        hash_md5 = hashlib.md5()
        with open(file_save_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


    def check_md5(self, index=-1):
        '''
        checks the md5 sum of file with index
        by default it checks the last downloaded file
        '''
        check_file = self.downloaded_df.iloc[index].filename
        md5_sum = self.allfiles_df[self.allfiles_df.filename == check_file].md5sum.iloc[0]
        other_md5_sum = self.md5sum(check_file)
        return md5_sum == other_md5_sum


    def filesize(self, filename):
        return_size = self.allfiles_df.loc[self.allfiles_df.filename == filename].iloc[0]['size']
        return int(return_size)


if __name__ == '__main__':
    '''
    Usage python3 dload.py
    '''
    ## Default values
    mountpoint = '/mnt/arXiv_src/'
    allfiles = 'allfiles3.csv'
    doun = 'downloaded_log.csv'
    error_log = 'error_dload.log'
    logging.basicConfig(filename='../error_log_dload.log', 
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(message)s')
    D = DownloadMan(mountpoint, allfiles, doun, error_log)
    while True:
        try:
            D.get_next()
        except IndexError:
            print('No more files to download')
            break
    sys.exit(0)
