import shutil
import os

def import_data(src,dst):
    shutil.copytree(src, dst,dirs_exist_ok=True)

import_data('/projectnb/eb-general/shared_data/data/processed/landsat/cloudfree_tiles',f'{os.getcwd()}/landsat')
import_data('/projectnb/eb-general/shared_data/data/processed/mlhfi/mlhfi2.0/predicted/','saved/predictions')
import_data('/projectnb/eb-general/shared_data/data/processed/mlhfi/mlhfi2.0/labels','data/')