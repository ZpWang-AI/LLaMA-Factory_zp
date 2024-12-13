from __head import *

# === TODO: prepare gpu ===
# CUDA_CNT = 1  
# cuda_id = CUDAUtils.set_cuda_visible(
#     target_mem_mb=20000,
#     cuda_cnt=1,
#     device_range=None,
# )
# CUDA_ID = GPUManager.set_cuda_visible(target_mem_mb=24000, cuda_cnt=CUDA_CNT)
# CUDA_ID = CustomArgs().prepare_gpu(target_mem_mb=10500, gpu_cnt=CUDA_CNT) 

from main import *


if __name__ == "__main__":
    main = LLaMA()

    def _dataset_config():
        IDRRdfs = IDRRDataFrames(
            data_name='pdtb3',
            data_level='top',
            data_relation='Implicit',
            data_path=r''
        )

        main.trainset_config = IDRRDatasetConfig(**IDRRdfs.json_dic)
        main.trainset_config.data_split = 'train'
        main.trainset_config.prompt = {
            "instruction": "Figure out the relation between the pair of arguments. The answer should be one of (Expansion, Temporary, Contingency and Comparison).\n\nThe first argument is\n\n{arg1}\n\nThe second argument is\n\n{arg2}",
            "input": '',
            "output": '{label11}',
            "system": "",
            "history": [],
        }
        main.trainset_config.desc = '_local_test'

    def _trainer_config():
        trainer_config = main.trainer_config

        trainer_config.model_name_or_path = '/home/user/test/pretrained_model/Llama-3-8B-Instruct'
        trainer_config.adapter_name_or_path

        trainer_config.do_train = True
        trainer_config.predict_with_generate = False
        trainer_config.lora_rank = 8
        trainer_config.lora_alpha = 16

        trainer_config.template = 'llama3'
        trainer_config.cutoff_len = 2048
        trainer_config.max_samples
        trainer_config.overwrite_cache = True
        trainer_config.preprocessing_num_workers = 16

        trainer_config.logging_steps = 10
        trainer_config.save_steps = 1000
        trainer_config.plot_loss = True
        trainer_config.overwrite_output_dir = True

        trainer_config.gradient_accumulation_steps = 8
        trainer_config.learning_rate = 5e-5
        trainer_config.num_train_epochs = 5
        trainer_config.warmup_ratio = 0.1
        trainer_config.bf16 = True
    
    def _extra_setting():
        extra_setting = main.extra_setting
        extra_setting.rest_mem_mb = 10**9
        extra_setting.wait_befor_start = 3
        extra_setting.output_scores = False
        extra_setting.do_dev = False

    _dataset_config()
    _trainer_config()
    _extra_setting()
    main.output_dir = ROOT_DIR/'exp_space'/'Inbox'
    main.desc = '_local_test'
    main._version_info_list = [
        Datetime_(), main.desc, 
        f'bs{main.trainer_config.per_device_eval_batch_size}~{main.trainer_config.gradient_accumulation_steps}_lr{main.trainer_config.learning_rate}_ep{main.trainer_config.num_train_epochs}'
    ]
    
    main.start(is_train=True, target_mem_mb=20000)


        