import logging
import os
import pdb
 
def get_logger(filename, verbosity=1, name=None):
    log_path = os.path.dirname(filename)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(filename):
        open(filename, "w+")
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger