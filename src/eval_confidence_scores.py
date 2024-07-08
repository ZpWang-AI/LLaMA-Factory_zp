from utils_zp import *
from IDRR_data import IDRRDataFrames

import numpy as np


class ConfidenceScoresEvaluator:
    @staticmethod
    def scores_to_confidence_scores(scores):
        return np.mean(scores)
        
    def __init__(self, dfs:IDRRDataFrames, split, target_res_dir) -> None:
        df = dfs.get_dataframe(split=split)
        target_res_dir = path(target_res_dir)
        self.scores_dict = build_dict_from_df_or_dicts(
            load_json(target_res_dir/'generated_scores.jsonl'),
            key_col_name='label', val_col_name='scores'
        )
        pred_dict = build_dict_from_df_or_dicts(
            load_json(target_res_dir/'generated_predictions.jsonl'),
            key_col_name='label', val_col_name='predict'
        )
        gt_dict = build_dict_from_df_or_dicts(
            df, key_col_name='data_id', val_col_name='label11'
        )
        gt_dict = dict((str(k),v)for k,v in gt_dict.items())
        processed_res = postprocess_generation_res_to_lid(
            pred=pred_dict, gt=gt_dict, label_list=dfs.label_list,
            match_strategy='first exists', 
            lower_results=True,
        )
        self.pred_dict = processed_res['pred']
        self.gt_dict = processed_res['gt']
        assert sorted(pred_dict.keys())==sorted(gt_dict.keys())==sorted(self.scores_dict.keys())
        print(processed_res['label_list'])
        # assert len(pred_dict)==len(gt_dict)==len(self.scores_dict)

    def get_target_confidence_score_correctness(self, target):
        conf_score = []
        correctness = []
        for data_id in self.gt_dict:
            if self.pred_dict[data_id] == target:
                correctness.append(self.pred_dict[data_id] == self.gt_dict[data_id])
                conf_score.append(ConfidenceScoresEvaluator.scores_to_confidence_scores(
                    self.scores_dict[data_id]
                ))
        return {
            'conf_score': conf_score,
            'correctness': correctness,
        }
    
    def draw_confidence_score_acc(self, conf_score, correctness, png_path):
        from scipy.ndimage import gaussian_filter1d
        conf_score, correctness = zip(*sorted(zip(conf_score,correctness), reverse=True))
        acc = []
        right = 0
        for pid, p in enumerate(correctness):
            if p:
                right += 1
            acc.append(right/(pid+1))
        cnt = len(acc)
        print(cnt)
        acc = gaussian_filter1d(acc, sigma=1)
        plt.figure(figsize=(8, 4.8))
        # print(acc)
        # x = conf_score
        # plt.gca().invert_xaxis()
        x = range(cnt)
        # plt.plot(conf_score, acc, marker='o', linestyle='-')
        # plt.plot(range(len(acc),0,-1),acc)
        plt.plot(x,acc, )
        for p in range(0,cnt,cnt//15):
            plt.text(p, acc[p], f'{conf_score[p]:.2f}', 
                     horizontalalignment='center', 
                     verticalalignment='bottom')
            plt.plot(p,acc[p], marker='o')
        plt.xlabel('sample_num')
        plt.ylabel('acc')
            # print(p, acc[p], conf_score[cnt-1-p])
        plt.savefig(png_path)
        plt.close()
    
    
if __name__ == '__main__':
    score_evalor = ConfidenceScoresEvaluator(
        dfs=IDRRDataFrames(
            data_name='pdtb3', data_level='top', data_relation='Implicit',
            data_path='/home/qwe/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv'
        ),
        split='test',
        target_res_dir='/home/qwe/test/zpwang/LLaMA/exp_space/2024-07-06-15-09-34.main_distill_all_thp.ckpt8000.bs1*8_lr0.0001_ep5',
    )
    for tar in range(4):
        score_evalor.draw_confidence_score_acc(
            **score_evalor.get_target_confidence_score_correctness(tar),
            png_path=f'/home/qwe/test/zpwang/LLaMA/exp_space/2024-07-06-15-09-34.main_distill_all_thp.ckpt8000.bs1*8_lr0.0001_ep5/conf_score_correctness_curve_{tar}.png'
        )