import os
import subprocess
import hashlib
import xml.etree.ElementTree as ET
import pandas as pd
import sys

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
        return self.allfiles_df[~idx].iloc[0]

    def log_error(self, err_message):
        with open(self.error_log_path, 'a') as efile:
            efile.write(err_message + '\n')
        return None


    def get(self, filename):
        file_save_path = os.path.join(self.mountpoint, filename)
        file_s3_path = self.s3_url + filename
        P = subprocess.run(
                ['/bin/s3cmd', 'get', '--requester-pays', 
                    file_s3_path, file_save_path],
                   stderr=subprocess.PIPE,
                   stdout=subprocess.PIPE)
        if P.returncode:
            print('Error Downloading file: %s'%filename)
            #with open(self.error_log_path, 'a') as efile:
            #    efile.write(str(P.stderr) + '\n')
            self.log_error(str(P.stderr))

        else:
            print('Download of file %s successful'%filename)
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
        return_bool = md5_sum == other_md5_sum
        if not return_bool:
            self.log_error('The md5 sum of the the file %s does not match'%check_file)
        return return_bool


    def filesize(self, filename):
        return_size = self.allfiles_df.loc[self.allfiles_df.filename == filename].iloc[0]['size']
        return int(return_size)



if __name__ == "__main__":
    mountpoint = sys.argv[1]
    allfiles = 'allfiles.csv'
    doun = 'downloaded_log.csv'
    error_log = 'error.log'
    D = DownloadMan(mountpoint, allfiles, doun, error_log)
    keep_going = 0
    while keep_going == 0:
        print('starting a download')
        keep_going = D.get_next()
        if D.check_md5():
            pass
        else:
            break




