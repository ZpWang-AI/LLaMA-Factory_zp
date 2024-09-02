from utils_zp import ExpArgs, make_path, load_json
from utils_zp.common_import import *


class ExtraSetting(ExpArgs):
    def __init__(self) -> None:
        self.rest_mem_mb = 1000000
        self.wait_befor_start = 3
        
        self.output_scores = True
        self.do_dev = False
    

class LLaMAFit(ExpArgs):
    def __init__(self) -> None:
        # model
        self.model_name_or_path = '/home/user/test/pretrained_model/Llama-3-8B-Instruct'
        self.adapter_name_or_path = None
        
        # method
        self.stage = 'sft'
        self.do_train = True
        self.predict_with_generate = False
        self.finetuning_type = 'lora'
        self.lora_target = 'all'
        self.lora_rank = 8
        self.lora_alpha = 16

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
        self.fp16 = True
        
        # eval
        self.val_size = 0
        self.per_device_eval_batch_size = 1
        self.eval_strategy = 'steps'
        self.eval_steps = 1000
        
        # additional_setting
        self._extra_setting = ExtraSetting()
    
    def start(self, cuda_id, llamafactory_path='.'):
        os.chdir(llamafactory_path)
        
        self.model_name_or_path = path(self.model_name_or_path)
        dataset_info_path = path('data')/'dataset_info.json'
        # build_dataset_info_path = path('data')/'build_dataset_info.json'
        self.output_dir = path(self.output_dir)/self.version
        assert self.model_name_or_path.exists()
        assert self.dataset in load_json(dataset_info_path)
        make_path(dir_path=self.output_dir)

        if self._extra_setting.do_dev:
            self.eval_steps = self.save_steps
        else:
            self.eval_steps = 10**10
        
        arg_yaml_path = self.output_dir/'fit_arg.yaml'
        self.dump_yaml(arg_yaml_path)
        self._extra_setting.dump_json(self.output_dir/'extra_setting.json')
        
        cmd = f"""
        CUDA_VISIBLE_DEVICES={cuda_id} llamafactory-cli train {arg_yaml_path}
        """.strip()
        os.system(cmd)
        pass


if __name__ == '__main__':
    sample = LLaMAFit()
    sample.start(0, '/home/user/test/zpwang/LLaMA-Factory')