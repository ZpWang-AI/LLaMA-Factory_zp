from __head import *
from main import *


if __name__ == "__main__":
    dfs = IDRRDataFrames(
        data_name='pdtb3',
        data_level='top',
        data_relation='Implicit',
        data_path=ROOT_DIR/'data'/'used'/'pdtb3.p2.csv'
    )
    trainset_config = IDRRDatasetConfig(
        data_split='train',
        prompt={
            "instruction": "Figure out the relation between the pair of arguments. The answer should be one of (Expansion, Temporary, Contingency and Comparison).\n\nThe first argument is\n\n{arg1}\n\nThe second argument is\n\n{arg2}",
            "input": '',
            "output": '{label11}',
            "system": "",
            "history": [],
        },
        desc='_local_test',
        **dfs.arg_dic,
    )

    model_path = path('/public/home/hongy/pretrained_models/Llama-3-8B-Instruct').resolve()
    # print(model_path)
    # print(model_path.exists())
    trainer_config = LLaMALoraSFTConfig(
        model_name_or_path=model_path,
        # adapter_name_or_path=

        do_train=True,
        predict_with_generate=False,
        lora_rank=8,
        lora_alpha=16,

        template='llama3',
        cutoff_len=2048,
        max_samples=40, # ===
        overwrite_cache=True,
        preprocessing_num_workers=16,

        logging_steps=30,
        save_steps=300,
        plot_loss=True,
        overwrite_output_dir=True,

        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=5,
        warmup_ratio=0.1,
        bf16=False,
        fp16=True,

        eval_steps=1000,
    )
    
    extra_setting = ExtraSetting(
        rest_mem_mb=10**9,
        wait_befor_start=3,
        output_scores=False,
        do_dev=False,
    )

    main = LLaMA(
        trainset_config=trainset_config,
        trainer_config=trainer_config,
        extra_setting=extra_setting,
        output_dir=ROOT_DIR/'exp_space'/'Inbox',
        desc='_local_test',
    )
    main._version_info_list = [
        Datetime_().format_str(2), main.desc, 
        f'bs{main.trainer_config.per_device_train_batch_size}-{main.trainer_config.gradient_accumulation_steps}_lr{main.trainer_config.learning_rate}_ep{main.trainer_config.num_train_epochs}'
    ]
    
    main.start(is_train=True, target_mem_mb=1000)


        