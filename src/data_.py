from _head import *


@config_args
@dataclass
class IDRRDatasetConfig:
    '''
    config = IDRRDatasetConfig() \\
    use config.start() to create dataset

    dataset_name = config.version

    auto update files in LLaMA-Factory/data:
    - dataset_info.json
    - target_file (example: IDRR/pdtb3.top/desc/train.json)
    - target_config_file (example: IDRR/pdtb3.top/desc/train.config.json)

    to remove data
    - del target_file and target_config_file
    - IDRRDatasetConfig.update_dataset_info()
    '''

    # ========== dataframes ==================
    part1:str = 'dataframes'
    data_name:str = 'pdtb3'
    data_level:str = 'top'
    data_relation:str = 'Implicit'
    data_path:str = ''
    data_split:str = 'train'
    
    # ========== base setting ================
    part2:str = 'base setting'
    prompt:dict = field(
        default_factory=lambda: {
            "instruction": "Figure out the relation between the pair of arguments. The answer should be one of (Expansion, Temporary, Contingency and Comparison).\n\nThe first argument is\n\n{arg1}\n\nThe second argument is\n\n{arg2}",
            "input": '',
            "output": '{label11}',
            "system": "",
            "history": []
        }
    )
    
    desc:str = '_test'
    # max_seq_length = 2048

    @property
    def version(self):
        info_list = [
            self.data_name,
            self.data_level,
            self.desc,
            self.data_split,
        ]
        return '.'.join(map(str, info_list))
        # return '.'.join(info_list).replace('-', '_')
    
    @property
    def target_file(self):
        return path(
            ROOT_DIR, 'data', 'IDRR',
            f'{self.data_name}.{self.data_level}',
            self.desc, self.data_split + '.json'
        )

    def start(self, update_dataset_info:bool=True):
        # check datapath
        self.data_path:path = path(self.data_path)
        assert self.data_path.exists()
        
        # check target_file
        target_file = self.target_file
        target_config_file = target_file.parent / (self.data_split + '.config.json')
        if target_file.exists():
            print(f'> Dataset "{self.version}" has been created')
            return

        # create dataset
        # get dataframes
        dfs = IDRRDataFrames(
            data_name=self.data_name,
            data_level=self.data_level,
            data_relation=self.data_relation,
            data_path=self.data_path,
        )
        # fill prompt
        filled_prompts = PromptFiller(
            df=dfs.get_dataframe(self.data_split), prompt=self.prompt,
        ).list
        # filter samples without output
        processed_data = [
            d for d in filled_prompts if d['output'] 
        ]
        
        # if self.max_seq_length:
        #     def clip_func(piece_of_data):
        #         for k, v in piece_of_data.items():
        #             if len(v) > self.max_seq_length:
        #                 piece_of_data[k] = v[:self.max_seq_length]
        #         return piece_of_data
        #     processed_data = list(map(clip_func, processed_data))
            
        if not processed_data:
            print(f'> "{self.version}" has no data')
            return
        
        make_path(file_path=target_file)
        auto_dump(processed_data, target_file)
        auto_dump(self.arg_dic, target_config_file)
        if update_dataset_info:
            self.update_dataset_info(False)

        print(f'> Succeed adding "{self.version}"')
        
    @staticmethod
    def update_dataset_info(print_info:bool=True):
        """
        staticmethod func

        update LLaMA-Factory.data.dataset_info.json \\
        by files endswith `.config.json`
        """
        IDRR_data_dir = LLAMA_FACTORY_DIR/'data'/'IDRR'
        make_path(IDRR_data_dir)

        dataset_info = {}
        for dirpath, dirnames, filenames in os.walk(IDRR_data_dir):
            for filename in filenames:
                if filename.endswith('.config.json'):
                    cfile = path(dirpath, filename)
                    config = IDRRDatasetConfig(**auto_load(cfile))
                    dataset_info[config.version] = {
                        'file_name': str(config.target_file),
                        # "columns": {
                        #     "prompt": "instruction",
                        #     "query": "input",
                        #     "response": "output",
                        #     "system": "system",
                        #     "history": "history"
                        # }
                    }
        
        auto_dump(dataset_info, LLAMA_FACTORY_DIR/'data'/'dataset_info.json')
        if print_info:
            print(f'> dataset_info.json updated')



# arg1 arg2 conn1 conn2 
# conn1sense1 conn1sense2 conn2sense1 conn2sense2
if __name__ == '__main__':
    IDRRDatasetConfig().format_part_in_file(__file__)
    exit()
    sample = IDRRDatasetConfig()
    sample.data_name = 'pdtb3'
    sample.data_level = 'top'
    sample.data_relation = 'Implicit'
    sample.data_path = ROOT_DIR/'data'/'used'/'pdtb3.p2.csv'
    sample.prompt = {
        'instruction': 'Arg1:\n{arg1}\n\nArg2:\n{arg2}',
        'output': '{label11id}',
    }
    sample.desc = '_local_test'
    sample.start()
    # sample.update_dataset_info()

