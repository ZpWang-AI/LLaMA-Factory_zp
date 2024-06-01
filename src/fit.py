import os


class Main:
    def __init__(self) -> None:
        self.cuda = '0'
        self.description = ''
        self.env_name = 'zpwang_main'
        self.exp_root_path = ''
        self.fp16 = True
        self.bf16 = False
        
        self.dataset = ''
        self.train_dataset = self.dataset+'_train'
        self.eval_dataset = self.dataset+'_test'
        
        self.model_path = ''
        self.template = 'qwen'
        self.lora_target = 'c_attn'
        
        self.epoch = 2
        self.batch_size = 4
        self.grad_acc_step = 4
        self.lr = 5e-4
        self.save_steps = 16000//(self.batch_size*self.grad_acc_step)
        self.logging_steps = 10
        self.lr_scheduler_type = 'cosine'
        
        self.max_test_samples = str(10**7)

    def redirect_output(self, cmd, output_path):
        return f'{cmd} > {output_path} 2>&1'

    def nohup_cmd(self, cmd,):
        return f'nohup {cmd} &'
    
    def train_cmd(self):
        return f"conda run -n {self.env_name} CUDA_VISIBLE_DEVICES=${self.cuda} python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ${{{self.model_path}}} \
    --dataset ${{{self.train_dataset}}} \
    --template ${{{self.template}}} \
    --finetuning_type lora \
    --lora_target ${{{self.lora_target}}} \
    --output_dir ${{{self.output_dir}}}/output \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size ${{{self.batch_size}}} \
    --gradient_accumulation_steps ${{{self.grad_acc_step}}} \
    --lr_scheduler_type ${{{self.lr_scheduler_type}}} \
    --logging_steps ${{{self.logging_steps}}} \
    --save_steps ${{{self.save_steps}}} \
    --learning_rate ${{{self.lr}}} \
    --num_train_epochs ${{{self.epoch}}} \
    --plot_loss \
    --fp16"
    
    def eval_cmd(self):
        return f"CUDA_VISIBLE_DEVICES=${{{self.cuda}}} python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ${{{self.model_path}}} \
    --adapter_name_or_path ${{{self.adapter_path}}}/output \
    --dataset ${{{self.eval_dataset}}} \
    --template ${{{self.template}}} \
    --finetuning_type lora \
    --output_dir ${{{self.output_dir}}}/output \
    --per_device_eval_batch_size 1 \
    --max_samples ${{{self.max_test_samples}}} \
    --predict_with_generate \
    --fp16"

    def main_iteration(self):
        cmd = [
            self.train_cmd(),
            self.eval_cmd(),
            self.eval_cmd(),
        ]
        cmd = '\n'.join(cmd)
        cmd = f'nohup sh -c "{cmd}"'
        print(os.popen(cmd=cmd).read())