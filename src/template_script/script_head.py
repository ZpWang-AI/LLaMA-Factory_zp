# ===== prepare server_name, root_fold =====
SERVER_NAME = 't2s'
if SERVER_NAME in ['cu13_', 'northern_']:
    ROOT_DIR = '/data/zpwang/IDRR_ConnT5/'
    PRETRAINED_MODEL_DIR = '/data/zpwang/pretrained_models/'
elif SERVER_NAME == 'cu12_':
    raise 
    ROOT_DIR = '/home/zpwang/IDRR/'
elif SERVER_NAME == 'SGA100':
    ROOT_DIR = '/public/home/hongy/zpwang/IDRR_ConnT5/'
    PRETRAINED_MODEL_DIR = '/public/home/hongy/pretrained_models/'
elif SERVER_NAME == 't2s':
    ROOT_DIR = '/home/qwe/test/zpwang/LLaMA/'
    PRETRAINED_MODEL_DIR = '/home/qwe/test/pretrained_model/'
else:
    raise Exception('wrong ROOT_DIR')

from utils_zp.common_import import *
from utils_zp import set_process_title
set_process_title('llama', add_time=True)

BRANCH = 'main'
CODE_SPACE = ROOT_DIR+'src/'
DATA_SPACE = ROOT_DIR+'data/'
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CODE_SPACE)