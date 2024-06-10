import os

from pathlib import Path as path
from setproctitle import setproctitle

from utils import AttrDict, ExpArgs, make_path, load_json, dump_json


def fill_with_delimiter(s:str):
    return f'{"="*10} {s} {"="*(30-len(s))}' if not s.startswith('='*10) else s


class Fit(ExpArgs):
    def __init__(self) -> None:
        # ========== 'base setting' ================
        self.part1 = 'base setting'
        self.description = 'test'
        self.save_ckpt = False
        self.cuda_cnt = 1
        
        # ========== 'file path' ===================
        self.part2 = 'file path'
        self.root_dir = '/home/qwe/test/zpwang/LLaMA/'
        self.llama_factory_dir = '/home/qwe/test/zpwang/LLaMA/LLaMA-Factory/'
        self.exp_space_dir = '/home/qwe/test/zpwang/LLaMA/exp_space/'
        self.log_dir = '/home/qwe/test/zpwang/LLaMA/logs/'
        
        # ========== 'data' ========================
        self.part3 = 'data'
        self.dataset_name = ''
        
        self.dataset_info = {}
        self.data_desc = ''
        
        # ========== 'model' =======================
        self.part4 = 'model'
        self.model_name = ''
        self.model_path = ''
        
        self.model_template = 'qwen'
        self.model_lora_target = 'c_attn'
        
        # ========== 'trainer' =====================
        self.part5 = 'trainer'
        self.env_name = 'zpwang_main'
        self.fp16 = True
        self.bf16 = False
        
        # ========== 'epoch, batch, step' ==========
        self.part6 = 'epoch, batch, step'
        self.epoch = 2
        self.batch_size = 4
        self.grad_acc_step = 4
        self.lr = 5e-4
        self.lr_scheduler_type = 'cosine'
        self.max_test_samples = 10**7
        self.save_steps = 16000//(self.batch_size*self.grad_acc_step)
        self.logging_steps = 10
        
        # ========== 'additional details' ==========
        self.part7 = 'additional details'
        self.cuda_id = ''
        self.server_name = ''
        self.create_time = ''
    
    @property
    def version(self):
        model_name = self.model_name.replace('-', '')
        return '.'.join([
            self.create_time,
            self.data_desc,
            self.description,
            f'ep{self.epoch}_bs{self.batch_size}_lr{self.lr}_{model_name}'
        ])
    
    def check_self(self):
        def check_path():
            self.root_dir = path(self.root_dir)
            self.llama_factory_dir = path(self.llama_factory_dir)
            self.exp_space_dir = path(self.exp_space_dir)
            self.log_dir = path(self.log_dir)
            assert path(self.root_dir).exists()
            assert path(self.llama_factory_dir).exists()
            make_path(self.exp_space_dir/self.version)
            make_path(self.log_dir)
        
        def get_build_dataset_info():
            info_path = self.llama_factory_dir/'data'/'build_dataset_info.json'
            self.dataset_info = load_json(info_path)[self.dataset_name]
        
        def get_model_arg():
            if self.model_name == 'qwen':
                pass
            elif self.model_name == 'llama2':
                pass
            raise 'wrong model_name'
        
        self.format_part()
        if not self.create_time:
            self.set_create_time()
        check_path()
        get_build_dataset_info()
        get_model_arg()
        
        pass
    
    def add_fp16_bf16(self, cmd):
        assert not (self.fp16 and self.bf16)
        if self.fp16:
            cmd += ' --fp16'
        elif self.bf16:
            cmd += ' --bf16'
        return cmd
    
    def redirect_output(self, cmd, log_path):
        return f'{cmd} > {log_path} 2>&1'

    def nohup_cmd(self, cmd,):
        return f'nohup {cmd} &'
    
    def train_cmd(self, train_dataset, output_dir):
        cmd = f'''
conda run -n {self.env_name} \
CUDA_VISIBLE_DEVICES={self.cuda_id} python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path {self.model_path} \
    --dataset {train_dataset} \
    --template {self.model_template} \
    --finetuning_type lora \
    --lora_target {self.model_lora_target} \
    --output_dir {output_dir} \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size {self.batch_size} \
    --gradient_accumulation_steps {self.grad_acc_step} \
    --lr_scheduler_type {self.lr_scheduler_type} \
    --logging_steps {self.logging_steps} \
    --save_steps {self.save_steps} \
    --learning_rate {self.lr} \
    --num_train_epochs {self.epoch} \
    --plot_loss
        '''.strip()
        cmd = self.add_fp16_bf16(cmd)
        return cmd
    
    def eval_cmd(self, adapter_path, eval_dataset, output_dir):
        cmd = f"""
conda run -n {self.env_name} \
CUDA_VISIBLE_DEVICES={self.cuda_id} python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path {self.model_path} \
    --adapter_name_or_path {adapter_path} \
    --dataset {eval_dataset} \
    --template {self.model_template} \
    --finetuning_type lora \
    --output_dir {output_dir} \
    --per_device_eval_batch_size 1 \
    --max_samples {self.max_test_samples} \
    --predict_with_generate
        """.strip()
        cmd = self.add_fp16_bf16(cmd)
        return cmd
    
    def main_iteration(self, train=True, dev=True, test=True):
        self.check_self()
        setproctitle('zpwang-llama.'+self.version)
        os.chdir(self.llama_factory_dir)
        
        cmd = []
        # if train:
        #     cmd.append(self.train_cmd(
        #         train_dataset=self.dataset+'_train',
        #         output_dir='',
        #     ))
        # if dev:
        #     for _ in range():
        #         cmd.append(self.eval_cmd(
        #             adapter_path=,
        #             eval_dataset=,
        #             output_dir=,
        #         ))
        # if test:
        #     cmd.append(self.eval_cmd(
        #         adapter_path=,
        #         eval_dataset=,
        #         output_dir=,
        #     ))
        
        cmd = '\n\n'.join(cmd)
        log_path = path(self.log_dir)/f'{self.version}.log'
        cmd = f'nohup sh -c "{cmd}" > {log_path} 2>&1 &'
        print(cmd)
        # print(os.popen(cmd=cmd).read())