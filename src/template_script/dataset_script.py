from script_head import *

# ===== import =====
from build_dataset import BuildDataset

# argX connX connXsenseY labelXY labelXYid 
class PromptMaker:
    @staticmethod
    def prompt_alpaca():
        train_prompt = {
            "instruction": """
Argument 1:
{arg1}

Argument 2:
{arg2}

{subtext}

Question: What is the discourse relation between Argument 1 and Argument 2?
A. Comparison
B. Contingency
C. Expansion
D. Temporal

Answer:""".strip(),
            "input": "",
            "output": "{label11}",
            "system": "",
            "history": []
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

    # ==================
    sample.desc = 'main_subtext_distill'
    sample.data_path = '/home/user/test/zpwang/IDRR_data/data/dataBuild/pdtb3_top_implicit.csv'
    # ==================

    sample.llama_factory_dir = '/home/user/test/zpwang/LLaMA-Factory'
    
    sample.prompt = prompt
    sample.max_seq_length = 10**20
    
    sample.start()
    
    # sample.rebuild_dataset_info(llama_factory_dir='/home/user/test/zpwang/LLaMA-Factory')