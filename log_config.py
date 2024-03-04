import logging
import os

def setup_logging(results_folder_path):
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    log_file_path = os.path.join(results_folder_path, 'application.log')

    logging.basicConfig(filename=log_file_path, 
                        filemode='a', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                        level=logging.INFO)