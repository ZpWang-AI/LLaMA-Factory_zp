# ===== prepare server_name, root_fold =====
SERVER_NAME = 't2s'
if SERVER_NAME in ['cu13_', 'northern_']:
    ROOT_DIR = '/data/zpwang/LLaMA/'
    PRETRAINED_MODEL_DIR = '/data/zpwang/pretrained_models/'
elif SERVER_NAME == 'cu12_':
    raise 
    ROOT_DIR = '/home/zpwang/IDRR/'
elif SERVER_NAME == 'SGA100':
    ROOT_DIR = '/public/home/hongy/zpwang/LLaMA/'
    PRETRAINED_MODEL_DIR = '/public/home/hongy/pretrained_models/'
elif SERVER_NAME == 't2s':
    ROOT_DIR = '/home/qwe/test/zpwang/LLaMA/'
    PRETRAINED_MODEL_DIR = '/home/qwe/test/pretrained_model/'
else:
    raise Exception('wrong ROOT_DIR')

from utils_zp.common_import import *

BRANCH = 'main'
CODE_SPACE = ROOT_DIR+'src/'
DATA_SPACE = ROOT_DIR+'data/used/'
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CODE_SPACE)

# from arguments import CustomArgs
from utils_zp.gpu_utils import GPUManager
# === TODO: prepare gpu ===
CUDA_CNT = 1  
CUDA_ID = GPUManager.set_cuda_visible(target_mem_mb=24000, cuda_cnt=CUDA_CNT)
# CUDA_ID = CustomArgs().prepare_gpu(target_mem_mb=10500, gpu_cnt=CUDA_CNT) 

from utils_zp import get_cur_time
from llama_fit import LLaMAFit


if __name__ == "__main__":
    main = LLaMAFit()
    main.model_name_or_path = '/home/qwe/test/pretrained_model/Llama-3-8B-Instruct'
    main.adapter_name_or_path = ''
    main.dataset = 'pdtb3.top.2024_06_11_21_41_36.base.clip2048'
    main.output_dir = '/home/qwe/test/zpwang/LLaMA/exp_space/'
    llamafactory_path = '/home/qwe/test/zpwang/LLaMA-Factory'
    desc = 'base'
    ckpt = 'final'
    
    main.per_device_eval_batch_size = 1
    main.gradient_accumulation_steps = 8
    main.learning_rate = 1e-4
    main.num_train_epochs = 5
    
    main.template = 'llama3'
    main.cutoff_len = 2048
    main.max_samples
    main.logging_steps = 10
    main.save_steps = 1000
    main.per_device_eval_batch_size = 1
    
    main.do_train = False
    main.predict_with_generate = True
    
    main._version_info_list = [
        get_cur_time(), desc, f'ckpt{ckpt}', 
        f'bs{main.per_device_eval_batch_size}*{main.gradient_accumulation_steps}_lr{main.learning_rate}_ep{main.num_train_epochs}'
    ]
    
    main.start(
        cuda_id=CUDA_ID,
        llamafactory_path=llamafactory_path
    )


        