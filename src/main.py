from _head import *
from utils_zp.cuda import *

from data_ import *


class ExtraSetting(ExpArgs):
    def __init__(self) -> None:
        self.rest_mem_mb = 1000000
        self.wait_befor_start = 3
        
        self.output_scores = True
        self.do_dev = False

class LLaMALoraSFTConfig(ExpArgs):
    def __init__(self, *args, **kwargs):
        # model
        self.model_name_or_path = '/home/user/test/pretrained_model/Llama-3-8B-Instruct'
        self.adapter_name_or_path = None  #
        
        # method
        self.stage = 'sft'
        self.do_train = True
        self.predict_with_generate = False  #
        self.finetuning_type = 'lora'
        self.lora_target = 'all'
        self.lora_rank = 8  #
        self.lora_alpha = 16  #

        # dataset
        self.dataset = 'pdtb3.top.2024_06_11_21_41_36.base.clip2048'
        self.template = 'llama3'
        self.cutoff_len = 2048
        self.max_samples = 10**10
        self.overwrite_cache = True
        self.preprocessing_num_workers = 16
        
        # output
        self.output_dir = '/home/user/test/zpwang/LLaMA/exp_space/test'
        self.logging_steps = 10
        self.save_steps = 1000
        self.plot_loss = True
        self.overwrite_output_dir = True
        
        # train
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 8
        self.learning_rate = 1.0e-4
        self.num_train_epochs = 5.0
        self.lr_scheduler_type = 'cosine'
        self.warmup_ratio = 0.1
        # self.fp16 = True
        self.bf16 = True
        self.ddp_timeout = 180000000
        
        # eval
        self.val_size = 0
        self.per_device_eval_batch_size = 1
        self.eval_strategy = 'steps'
        self.eval_steps = 1000


class MainConfig(ExpArgs):
    '''
    final result dir is `self.output_dir/self.version`

    result files:
    - trainer_config.yaml
    - main_config.json

    trainer_config.eval_steps would be set autoly

    cmd:
        `CUDA_VISIBLE_DEVICES=xx llamafactory-cli train {yaml_path}`
    '''
    def __init__(self) -> None:
        # 
        self.part1 = 'data'
        self.trainset_config: IDRRDatasetConfig = None
        self.devset_config: IDRRDatasetConfig = None
        self.testset_config: IDRRDatasetConfig = None

        self.part2 = 'trainer'
        self.trainer_config = LLaMALoraSFTConfig()
        self.extra_setting = ExtraSetting()

        self.part3 = 'additonal'
        self._version_info_list = []
        self.set_create_time()
        self.format_part()

    def start(self):
        cuda_id = CUDAUtils.set_cuda_visible(
            target_mem_mb=20000,
            cuda_cnt=1,
            device_range=None,
        )
        
        # prepare data
        for dataset_config in [self.trainset_config, self.devset_config, self.testset_config]:
            if dataset_config is not None:
                dataset_config.start(False)
        IDRRDatasetConfig.update_dataset_info(False)
        print('> data prepared\n')

        # config
        if self.extra_setting.do_dev:
            self.trainer_config.eval_steps = self.trainer_config.save_steps
        else:
            self.trainer_config.eval_steps = 10**10

        # path
        os.chdir(LLAMA_FACTORY_DIR)
        self.model_name_or_path = path(self.model_name_or_path)
        self.output_dir = path(self.output_dir)/self.version
        assert self.model_name_or_path.exists()
        assert not self.output_dir.exists()
        make_path(dir_path=self.output_dir)

        arg_yaml_path = self.output_dir/'trainer_config.yaml'
        auto_dump(self.trainer_config.json_dic, arg_yaml_path)
        # auto_dump(self.extra_setting, self.output_dir/'extra_setting.json')
        auto_dump(self, self.output_dir/'main_config.json')

        cmd = f"""
        CUDA_VISIBLE_DEVICES={cuda_id} llamafactory-cli train {arg_yaml_path}
        """.strip()
        os.system(cmd)
        pass


if __name__ == '__main__':
    sample = MainConfig()
    # sample.start(0, '/home/user/test/zpwang/LLaMA-Factory')
