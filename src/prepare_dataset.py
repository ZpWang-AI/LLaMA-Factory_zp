import json
import datetime

from typing import *
from copy import deepcopy
from pathlib import Path as path

from IDRR_data import DataFrame, PromptFiller
from utils import AttrDict


class DatasetArgs(AttrDict):
    def __init__(
        self,
        data_name:Literal['pdtb2', 'pdtb3', 'conll', None],
        label_level:Literal['level1', 'level2', 'raw'],
        relation:Literal['Implicit', 'Explicit', 'All'],
        data_path:str,
        prompt,
        version,

        max_seq_length:int=None,
        label_use_id=False,
        # train_multi_label:bool=False,
        # test_multi_label:bool=False,
        create_time=None,
    ) -> None:
        self.data_name = data_name
        self.label_level = label_level
        self.relation = relation
        self.data_path = data_path
        self.prompt = prompt
        self.version = version
        
        self.max_seq_length = max_seq_length
        self.label_use_id = label_use_id
        # self.train_multi_label = train_multi_label
        # self.test_multi_label = test_multi_label
        self.set_create_time(create_time=create_time)
        
        dataframe = DataFrame(
            data_name=data_name,
            label_level=label_level,
            relation=relation,
            data_path=data_path,
            label_use_id=label_use_id,
        )
        self.build_datasets(dataframe=dataframe)

    def dump_json(self, processed_data_name):
        json_path = path('/public/home/hongy/zpwang/LLaMA-Factory/data/dataset_args')
        json_path /= f'{processed_data_name}.json'
        self._dump_json(json_path=json_path, overwrite=True)
    
    def build_datasets(self, dataframe:DataFrame):
        processed_data_name = [
            self.data_name,
            self.label_level,
            self.create_time,
            self.version,
        ]
        if self.max_seq_length:
            processed_data_name.append(f'clip{self.max_seq_length}')
        
        processed_data_name = '_'.join(processed_data_name).replace('-', '_')
        
        self.build_single_dataset(
            processed_data=list(PromptFiller(
                df=dataframe.train_df,
                prompt=self.prompt,
            )),
            processed_data_name=processed_data_name+'_train'
        )
        
        for split in 'dev test blind_test'.split():
            self.build_single_dataset(
                processed_data=list(PromptFiller(
                    df=getattr(dataframe, f'{split}_df'),
                    prompt=self.prompt,
                    ignore=('reason',)
                )),
                processed_data_name=processed_data_name+'_'+split
            )
        
        self.dump_json(processed_data_name)
        
            
    def build_single_dataset(self, processed_data, processed_data_name):
        if self.max_seq_length:
            def filter_func(processed_datum):
                seq_len = sum(len(processed_datum[p])for p in 'instruction input system'.split())
                return seq_len < self.max_seq_length
            processed_data = list(filter(filter_func, processed_data))
            
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
