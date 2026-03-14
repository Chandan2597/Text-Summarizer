import os
import urllib.request as request
import zipfile
from TextSummarizer.logging import logger
from TextSummarizer.utils.common import get_size
from TextSummarizer.entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):

            headers = {'User-Agent': 'Mozilla/5.0'}
            req = request.Request(
                url=self.config.source_URL,
                headers=headers
            )

            with request.urlopen(req) as response, open(self.config.local_data_file, 'wb') as out_file:
                out_file.write(response.read())

            logger.info("File downloaded successfully!")

        else:
            logger.info("File already exists")  

        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)