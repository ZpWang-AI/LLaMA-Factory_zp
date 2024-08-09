import os, sys, json
import datetime

from typing import *
from copy import deepcopy
from pathlib import Path as path

from IDRR_data import IDRRDataFrames, PromptFiller
from utils_zp import AttrDict, ExpArgs, dump_json, load_json


class BuildDataset(ExpArgs):
    def __init__(self) -> None:
        # ========== 'data' ========================
        self.part1 = 'data'
        self.data_name = 'pdtb3'
        self.data_level = 'level1'
        self.data_relation = 'Implicit'
        self.data_split = ''

        # ========== 'path' ========================
        self.part2 = 'path'
        self.data_path = ''
        self.llama_factory_dir = '/home/user/test/zpwang/LLaMA/LLaMA-Factory'
        
        # ========== 'base setting' ================
        self.part3 = 'base setting'
        self.prompt = {
            "instruction": "Figure out the relation between the pair of arguments. The answer should be one of (Expansion, Temporary, Contingency and Comparison).",
            "input": 'The first argument is\n\n{arg1}\n\nThe second argument is\n\n{arg2}',
            "output": [
                "{conn1sense1}",
                "I don't know.",
            ],
            "system": "",
            "history": []
        }
        self.desc = 'test'
        self.max_seq_length = 2048
        
        # ========== 'additional info' =============
        self.part4 = 'additional info'
        self.set_create_time()

    @property
    def version(self):
        info_list = [
            self.data_name,
            self.data_level,
            self.create_time,
            self.desc,
        ]
        return '.'.join(info_list).replace('-', '_')
    
    def start(self):
        self.data_path = path(self.data_path)
        self.llama_factory_dir = path(self.llama_factory_dir)
        assert self.data_path.exists()
        assert self.llama_factory_dir.exists()
        self.format_part()
        
        dataframes = IDRRDataFrames(
            data_name=self.data_name,
            data_level=self.data_level,
            data_relation=self.data_relation,
            data_path=self.data_path,
        )
        
        if 'train' in self.prompt and 'pred' in self.prompt:
            train_prompt = self.prompt['train']
            pred_prompt = self.prompt['pred']
            self.build_single_dataset(
                processed_data=PromptFiller(
                    df=dataframes.train_df,
                    prompt=train_prompt,
                ).list,
                processed_data_name=self.version+'.train'
            )
            self.build_single_dataset(
                processed_data=PromptFiller(
                    df=dataframes.dev_df,
                    prompt=train_prompt,
                ).list,
                processed_data_name=self.version+'.dev'
            )
            self.build_single_dataset(
                processed_data=PromptFiller(
                    df=dataframes.test_df,
                    prompt=pred_prompt,
                ).list,
                processed_data_name=self.version+'.test'
            )
        elif 'pred' in self.prompt:
            assert self.data_split in 'train dev test all'.split()
            pred_prompt = self.prompt['pred']
            self.build_single_dataset(
                processed_data=PromptFiller(
                    df=dataframes.get_dataframe(split=self.data_split),
                    prompt=pred_prompt,
                ).list,
                processed_data_name=self.version+'.pred'
            )
        else:
            raise Exception('train&pred or pred in self.prompt')
        
        build_dataset_info_path = path(self.llama_factory_dir)/'data'/'build_dataset_info.json'
        if build_dataset_info_path.exists():
            build_dataset_info = load_json(build_dataset_info_path)
        else:
            build_dataset_info = {}
        build_dataset_info[self.version] = self.json
        dump_json(build_dataset_info, build_dataset_info_path, mode='w', indent=4)
            
    def build_single_dataset(self, processed_data:List[Dict[str, str]], processed_data_name):
        processed_data = [
            ppd for ppd in processed_data if ppd['output'] 
        ]
        
        if self.max_seq_length:
            def clip_func(piece_of_data):
                for k, v in piece_of_data.items():
                    if len(v) > self.max_seq_length:
                        piece_of_data[k] = v[:self.max_seq_length]
                return piece_of_data
            processed_data = list(map(clip_func, processed_data))
            
        if not processed_data:
            print(processed_data_name, 'has no data')
            return
        
        target_file = self.llama_factory_dir/'data'/f'{processed_data_name}.json'
        dump_json(processed_data, target_file, mode='w', indent=2)
        
        dataset_info_path:path = self.llama_factory_dir/'data'/'dataset_info.json'
        if dataset_info_path.exists():
            dataset_info = load_json(dataset_info_path)
        else:
            dataset_info = {}
        
        if processed_data_name not in dataset_info:
            dataset_info[processed_data_name] = {
                "file_name": str(target_file),
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history"
                }
            }
            dump_json(dataset_info, dataset_info_path, mode='w', indent=4)
            print('add data:', processed_data_name,)
            print(len(processed_data), '\n')
        else:
            print(processed_data_name, 'exists')

    @staticmethod
    def remove_dataset(dataset_name, llama_factory_dir):
        data_dir = path(llama_factory_dir)/'data'
        
        build_dataset_info_path = data_dir/'build_dataset_info.json'
        build_dataset_info = load_json(build_dataset_info_path)
        if dataset_name in build_dataset_info:
            del build_dataset_info[dataset_name]
        dump_json(build_dataset_info, build_dataset_info_path, mode='w', indent=4)

        dataset_info_path = data_dir/'dataset_info.json'
        dataset_info = load_json(dataset_info_path)
        for split in '.train .dev .test'.split()+['']:
            tar_dataset_name = dataset_name+split
            if tar_dataset_name in dataset_info:
                del dataset_info[tar_dataset_name]
        dump_json(dataset_info, dataset_info_path, mode='w', indent=4)

        for split in '.train .dev .test .pred'.split()+['']:
            tar_dataset_file:str = dataset_name+split+'.json'
            tar_dataset_file:path = data_dir/tar_dataset_file
            if tar_dataset_file.exists():
                os.remove(tar_dataset_file)
                print(f'remove {tar_dataset_file.stem}')

    @staticmethod
    def rebuild_dataset_info(llama_factory_dir):
        data_dir = path(llama_factory_dir)/'data'
        dataset_info = {}
        for file in sorted(os.listdir(data_dir)):
            dataset_name = path(file).stem
            dataset_info[dataset_name] = {
                'file_name': str(data_dir/file),
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history"
                }
            }
        dump_json(dataset_info, data_dir/'dataset_info.json')
        print('dump to', data_dir/'dataset_info.json')

# arg1 arg2 conn1 conn2 
# conn1sense1 conn1sense2 conn2sense1 conn2sense2

# BuildDataset.format_part_in_file(__file__)
if __name__ == '__main__':
    # BuildDataset.remove_dataset(
    #     dataset_name='pdtb3.top.2024_06_08_12_22_38.base.clip2048',
    #     llama_factory_dir='/home/user/test/zpwang/LLaMA/LLaMA-Factory'
    # )
    BuildDataset.rebuild_dataset_info('/home/user/test/zpwang/LLaMA-Factory')