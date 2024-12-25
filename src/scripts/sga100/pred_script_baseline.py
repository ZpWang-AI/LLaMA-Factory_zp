from __head import *
from main import *


if __name__ == "__main__":
    dfs = IDRRDataFrames(
        data_name='pdtb3',
        data_level='top',
        data_relation='Implicit',
        data_path=ROOT_DIR/'data'/'used'/'pdtb3.p2.csv'
    )
    testset_config = IDRRDatasetConfig(
        data_split='test',
        prompt={
            "instruction": '''
Argument 1:
{arg1}

Argument 2:
{arg2}

What's the discourse relation between Argument 1 and Argument 2?
A. Comparison
B. Contingency
C. Expansion
D. Temporal

'''.strip(),
            "input": '',
            "output": '{label11}',
            "system": "",
            "history": [],
        },
        desc='base_multi-choice',
        **dfs.arg_dic,
    )

    model_path = path('/public/home/hongy/pretrained_models/Llama-3-8B-Instruct').resolve()
    # ckpt_path = path('/public/home/hongy/zpwang/LLaMA-Factory_zp/exp_space/Inbox/2024-12-18_07-28-07._local_test.bs1-8_lr5e-05_ep5.succeed/src_output/checkpoint-16').resolve()
    # print(model_path)
    # print(model_path.exists())
    trainer_config = LLaMALoraSFTConfig(
        model_name_or_path=model_path,
        # adapter_name_or_path=ckpt_path,

        do_train=False,
        do_predict=True,
        predict_with_generate=True,
        lora_rank=8,
        lora_alpha=16,

        template='llama3',
        cutoff_len=2048,
        # max_samples=32, # ===
        overwrite_cache=True,
        preprocessing_num_workers=16,

        logging_steps=100,
        save_steps=1000,
        plot_loss=True,
        overwrite_output_dir=True,

        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        bf16=False,
        fp16=True,
    )
    
    extra_setting = ExtraSetting(
        rest_mem_mb=10**9,
        wait_befor_start=3,
        output_scores=False,
        do_dev=False,
    )

    target_mem_mb = 20000
    cuda_id = CUDAUtils.set_cuda_visible(
        target_mem_mb=target_mem_mb,
        cuda_cnt=1,
        device_range=None,
    )

    def predict(ckpt_path):
        trainer_config.adapter_name_or_path = ckpt_path
        CUDAUtils.get_free_cudas(
            target_mem_mb=target_mem_mb,
            cuda_cnt=1,
            device_range=[int(cuda_id)],
        )
        main = LLaMA(
            trainset_config=OneShotDatasetConfig(),
            testset_config=testset_config,
            trainer_config=trainer_config,
            extra_setting=extra_setting,
            output_dir=ROOT_DIR/'exp_space'/'Inbox',
            desc='_local_test',
            cuda_id=cuda_id,
        )
        main._version_info_list = [
            Datetime_().format_str(2), main.desc, 
            f'bs{main.trainer_config.per_device_train_batch_size}-{main.trainer_config.gradient_accumulation_steps}_lr{main.trainer_config.learning_rate}_ep{main.trainer_config.num_train_epochs}'
        ]
        
        main.start(bg_run=True)
        time.sleep(600)

    ckpt_dir = '/public/home/hongy/zpwang/LLaMA-Factory_zp/exp_space/Inbox/2024-12-23_13-12-16._baseline.bs1-8_lr0.0001_ep5.train'
    ckpt_dir = path(ckpt_dir) / 'src_output'
    ckpt_path_lst = [p for p in listdir_full_path(ckpt_dir) if p.is_dir()]
    ckpt_path_lst.append(ckpt_dir)
    for ckpt_path in ckpt_path_lst:
        predict(ckpt_path)

        