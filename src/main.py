from _head import *
from utils_zp.cuda import *

from data_ import *


@config_args
@dataclass
class ExtraSetting:
    rest_mem_mb:int = 1000000
    wait_befor_start:int = 3
    
    output_scores:bool = True
    # do_dev:bool = False

    # def __init__(self) -> None:
    #     self.rest_mem_mb = 1000000
    #     self.wait_befor_start = 3
        
    #     self.output_scores = True
    #     self.do_dev = False


@config_args
@dataclass
class LLaMALoraSFTConfig:
    # model
    model_name_or_path:str = '/home/user/test/pretrained_model/Llama-3-8B-Instruct'
    adapter_name_or_path:Optional[str] = None  #
    trust_remote_code:bool = True
    
    # method
    stage:str = 'sft'
    do_train:bool = True
    do_eval:bool = False
    do_predict:bool = False
    predict_with_generate:bool = False  #
    finetuning_type:str = 'lora'
    lora_target:str = 'all'
    lora_rank:int = 8  #
    lora_alpha:int = 16  #

    # dataset
    dataset_dir:str = LLAMA_FACTORY_DATASET_DIR
    dataset:str = 'pdtb3.top.2024_06_11_21_41_36.base.clip2048'
    eval_dataset:str = ''
    template:str = 'llama3'
    cutoff_len:int = 2048
    max_samples:int = 10**10
    overwrite_cache:bool = True
    preprocessing_num_workers:int = 16
    
    # output
    output_dir:str = '/home/user/test/zpwang/LLaMA/exp_space/test'
    logging_steps:int = 10
    save_steps:int = 1000
    plot_loss:bool = True
    overwrite_output_dir:bool = True
    
    # train
    per_device_train_batch_size:int = 1
    gradient_accumulation_steps:int = 8
    learning_rate:float = 1.0e-4
    num_train_epochs:float = 5.0
    lr_scheduler_type:str = 'cosine'
    warmup_ratio:float = 0.1
    bf16:bool = False
    fp16:bool = True
    ddp_timeout:int = 180000000
    
    # eval
    # val_size:float = 0.1
    per_device_eval_batch_size:int = 1
    eval_strategy:str = 'steps'
    eval_steps:int = 10**9


@config_args
@dataclass
class LLaMA:
    '''
    final result dir is `self.output_dir/self.version`

    result files:
    - trainer_config.yaml
    - main_config.json

    trainer_config.eval_steps would be set autoly

    cmd:
        `CUDA_VISIBLE_DEVICES=xx llamafactory-cli train {yaml_path}`
    '''

    # ========== data ========================
    part1:str = 'data'
    trainset_config:DatasetConfig = None
    testset_config:DatasetConfig = None

    # ========== trainer =====================
    part2:str = 'trainer'
    trainer_config:LLaMALoraSFTConfig = LLaMALoraSFTConfig()
    extra_setting:ExtraSetting = ExtraSetting()

    # ========== base ========================
    part3:str = 'base'
    output_dir:str = '.'
    desc:str = '_test'

    # ========== additonal ===================
    part4:str = 'additonal'
    cuda_id:str = None
    _version_info_list:list = None

    @property
    def version(self):
        return '.'.join(map(str, self._version_info_list))

    def start(self):
        # =======================================
        # prepare data
        self.trainset_config.start()
        self.testset_config.start()
        self.trainer_config.dataset = self.trainset_config.version
        self.trainer_config.eval_dataset = self.testset_config.version
        DatasetConfig.update_dataset_info(print_info=False)
        print('> data prepared\n')

        # # config
        # if self.extra_setting.do_dev:
        #     self.trainer_config.eval_steps = self.trainer_config.save_steps
        # else:
        #     self.trainer_config.eval_steps = 10**10

        # =======================================
        # path
        os.chdir(LLAMA_FACTORY_DIR)
        final_output_dir = path(self.output_dir) / self.version
        self.trainer_config.output_dir = final_output_dir / 'src_output'
        assert path(self.trainer_config.model_name_or_path).exists()
        assert not path(final_output_dir).exists()
        make_path(dir_path=self.trainer_config.output_dir)
        # log_path = self.trainer_config.output_dir/'nohup.log'

        arg_dic = self.trainer_config.arg_dic
        del arg_dic['create_time']
        arg_yaml_path = final_output_dir/'src_config.yaml'
        auto_dump(arg_dic, arg_yaml_path)
        # auto_dump(self.extra_setting, final_output_dir/'extra_setting.json')
        auto_dump(self.arg_dic, final_output_dir/'main_config.json')

        # =======================================
        # start running
        balancer = CUDABalancer(
            cuda_ids=[int(self.cuda_id)],
            rest_mem_mb=self.extra_setting.rest_mem_mb,
            wait_before_start=self.extra_setting.wait_befor_start,
        )
        balancer.start()
        
        cmd = (
            f'CUDA_VISIBLE_DEVICES={self.cuda_id} '
            f'llamafactory-cli train {arg_yaml_path}'
        )
        print(cmd+'\n')
        subprocess.run(
            cmd,
            shell=True,
            text=True,
        )

        balancer.close()


if __name__ == '__main__':
    sample = LLaMA()
    sample.format_part_in_file(__file__)
    print(sample)
    # sample.start(0, '/home/user/test/zpwang/LLaMA-Factory')
