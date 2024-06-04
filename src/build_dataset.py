import json
import datetime

from typing import *
from copy import deepcopy
from pathlib import Path as path

from IDRR_data import DataFrames, DataFrames2, PromptFiller
from utils import AttrDict, dump_json, load_json


class BuildDataset(AttrDict):
    def __init__(self) -> None:
        self.data_name = 'pdtb3'
        self.label_level = 'level1'
        self.relation = 'Implicit'
        self.data_path = ''
        self.llama_factory_dir = '/home/qwe/test/zpwang/LLaMA/LLaMA-Factory'
        
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
        self.set_create_time()

    @property
    def version(self):
        info_list = [
            self.data_name,
            self.label_level,
            self.create_time,
            self.desc,
            f'clip{self.max_seq_length}'
        ]
        return '.'.join(info_list).replace('-', '_')

    def dump_json(self, processed_data_name):
        json_path = path('/public/home/hongy/zpwang/LLaMA-Factory/data/dataset_args')
        json_path /= f'{processed_data_name}.json'
        self._dump_json(json_path=json_path, overwrite=True)
    
    def start(self):
        dataframes = DataFrames2(
            data_name=self.data_name,
            label_level=self.label_level,
            relation=self.relation,
            data_path=self.data_path,
        )
        
        train_prompt = self.prompt['train'] if 'train' in self.prompt else self.prompt
        eval_prompt = self.prompt['eval'] if 'eval' in self.prompt else self.prompt

        self.build_single_dataset(
            processed_data=list(PromptFiller(
                df=dataframes.train_df,
                prompt=train_prompt,
            )),
            processed_data_name=self.version+'_train'
        )
        
        for split in 'dev test'.split():
            self.build_single_dataset(
                processed_data=list(PromptFiller(
                    df=dataframes.get_dataframe(split=split),
                    prompt=eval_prompt,
                )),
                processed_data_name=self.version+'_'+split
            )
        
        build_dataset_info_path = path(self.llama_factory_dir)/'data'/'build_dataset_info.json'
        build_dataset_info = load_json(build_dataset_info)
        build_dataset_info[self.version] = self.to_json()
        dump_json(build_dataset_info)
        # self.dump_json(processed_data_name)
        
            
    def build_single_dataset(self, processed_data, processed_data_name):
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
        
        target_file = f'/public/home/hongy/zpwang/LLaMA-Factory/data/{processed_data_name}.json'
        with open(target_file, 'w', encoding='utf-8')as f:
            json.dump(processed_data, f, ensure_ascii=True, indent=2)
        
        dataset_info_file = path('/public/home/hongy/zpwang/LLaMA-Factory/data/dataset_info.json')
        if dataset_info_file.exists():
            with open(dataset_info_file, 'r', encoding='utf8')as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        
        if processed_data_name not in dataset_info:
            dataset_info[processed_data_name] = {
                "file_name": target_file,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                    "history": "history"
                }
            }
            with open(dataset_info_file, 'w', encoding='utf8')as f:
                json.dump(dataset_info, f, indent=4)
            print('add data:', processed_data_name, '\n')
        else:
            print(processed_data_name, 'exists')

# arg1 arg2 conn1 conn2 
# conn1sense1 conn1sense2 conn2sense1 conn2sense2
