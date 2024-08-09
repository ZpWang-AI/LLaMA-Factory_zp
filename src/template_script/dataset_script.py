from script_head import *

# ===== import =====
from build_dataset import BuildDataset

# arg1 arg2 conn1 conn2 
# conn1sense1 conn1sense2 conn2sense1 conn2sense2
class PromptMaker:
    
    @staticmethod
    def prompt_alpaca():
        train_prompt = {
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
        pred_prompt = dcopy(train_prompt)
        pred_prompt['output'] = '{data_id}'
        return {'train': train_prompt, 'pred': pred_prompt}
        
    
    
if __name__ == '__main__':
    prompt = PromptMaker.prompt_alpaca()
    print(prompt)
    print()
    sample = BuildDataset()
    sample.data_name = 'pdtb3'
    sample.data_level = 'top'
    sample.data_relation = 'Implicit'
    # sample.data_split = ''

    # ==================
    sample.desc = 'base'
    sample.data_path = '/home/user/test/zpwang/Trainer/data/used/pdtb3.p1.csv'
    # ==================

    sample.llama_factory_dir = '/home/user/test/zpwang/LLaMA-Factory'
    
    sample.prompt = prompt
    sample.max_seq_length = 1024
    
    sample.start()
    