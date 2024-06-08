# ===== prepare server_name, root_fold =====
SERVER_NAME = 't2s'
if SERVER_NAME in ['cu13_', 'northern_']:
    ROOT_DIR = '/data/zpwang/IDRR_ConnT5/'
    PRETRAINED_MODEL_DIR = '/data/zpwang/pretrained_models/'
elif SERVER_NAME == 'cu12_':
    raise 
    ROOT_DIR = '/home/zpwang/IDRR/'
elif SERVER_NAME == 'SGA100':
    ROOT_DIR = '/public/home/hongy/zpwang/IDRR_ConnT5/'
    PRETRAINED_MODEL_DIR = '/public/home/hongy/pretrained_models/'
elif SERVER_NAME == 't2s':
    ROOT_DIR = '/home/qwe/test/zpwang/Trainer'
    PRETRAINED_MODEL_DIR = '/home/qwe/test/pretrained_model/'
else:
    raise Exception('wrong ROOT_DIR')
ROOT_DIR = '/home/qwe/test/zpwang/LLaMA/'

import os, sys
from pathlib import Path as path

BRANCH = 'main'
CODE_SPACE = ROOT_DIR+'src/'
DATA_SPACE = ROOT_DIR+'data/'
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, CODE_SPACE)

# ===== import =====
from build_dataset import BuildDataset

# arg1 arg2 conn1 conn2 
# conn1sense1 conn1sense2 conn2sense1 conn2sense2
class PromptMaker:
    
    @staticmethod
    def prompt_alpaca():
        return {
            "instruction": 'Argument 1:\n{arg1}\n\nArgument 2:\n{arg2}\n\nQuestion: What is the discourse relation between Argument 1 and Argument 2?\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\n\nAnswer:',
            "input": "",
            "output": "{label11}",
            "system": "",
            # "system": "The task is to determine whether they have a temporal, comparative, contingency, or extensional relationship. This analysis should consider both implicit and explicit relationships.",
            "history": [
                # ['The first argument is\n\nHis recent appearance at the Metropolitan Museum, dubbed \"A Musical Odyssey,\" was a case in point\n\nThe second argument is\n\nIt felt more like a party, or a highly polished jam session with a few friends, than a classical concert', 'Expansion'],
                # ["The first argument is\n\nBach's \"Air\" followed\n\nThe second argument is\n\nMr. Stoltzman tied the composer in by proclaiming him \"the great improviser of the 18th century", 'Temporal'],
            ]
        }
        
    @staticmethod
    def prompt_preference():
        # {
        #     "instruction": "user instruction",
        #     "input": "user input",
        #     "output": [
        #         "chosen answer",
        #         "rejected answer"
        #     ]
        # }
        return {
            "instruction": "Figure out the relation between the pair of arguments. The answer should be one of (Expansion, Temporary, Contingency and Comparison).",
            "input": 'The first argument is\n\n{arg1}\n\nThe second argument is\n\n{arg2}',
            "output": [
                "{conn1sense1}",
                "I don't know.",
            ],
            "system": "",
            "history": []
        }
        
    @staticmethod
    def prompt_sharegpt():
        # {
        #     "conversations": [
        #         {
        #         "from": "human",
        #         "value": "user instruction"
        #         },
        #         {
        #         "from": "gpt",
        #         "value": "model response"
        #         }
        #     ],
        #     "system": "system prompt (optional)",
        #     "tools": "tool description (optional)"
        # }
        return {
            
        }
    
    
if __name__ == '__main__':
    prompt = PromptMaker.prompt_alpaca()
    print(prompt)
    print()
    sample = BuildDataset()
    sample.data_name = 'pdtb3'
    sample.data_level = 'top'
    sample.data_relation = 'Implicit'

    sample.data_path = '/home/qwe/test/zpwang/Trainer/data/used/pdtb3.p1.csv'
    sample.llama_factory_dir = ROOT_DIR+'LLaMA-Factory/'
    
    sample.prompt = prompt
    sample.max_seq_length = 2048

    sample.desc = 'base'
    
    sample.start()