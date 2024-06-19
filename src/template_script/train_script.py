from script_head import *

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
    main.adapter_name_or_path
    main.dataset = 'pdtb3.top.2024_06_11_21_41_36.base.clip2048'
    main.output_dir = '/home/qwe/test/zpwang/LLaMA/exp_space/'
    llamafactory_path = '/home/qwe/test/zpwang/LLaMA-Factory'
    desc = 'base'
    
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
        
    main.do_train = True
    main.predict_with_generate = False
    
    main._version_info_list = [
        get_cur_time(), desc, 
        f'bs{main.per_device_eval_batch_size}*{main.gradient_accumulation_steps}_lr{main.learning_rate}_ep{main.num_train_epochs}'
    ]
    
    main.start(
        cuda_id=CUDA_ID,
        llamafactory_path=llamafactory_path
    )


        