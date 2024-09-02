from utils_zp import *
from IDRR_data import IDRRDataFrames

import numpy as np
from sklearn.metrics import classification_report

from calculate_metric_by_scores import calculate_metric


class ConfidenceScoresEvaluator:
    def __init__(
        self, 
        dfs:IDRRDataFrames, split, 
        score_dir,
        rest_dir,
    ) -> None:
        self.dfs = dfs
        self.split = split
        df = dfs.get_dataframe(split=split)
        score_dir = path(score_dir)
        self.scores_dict = build_dict_from_df_or_dicts(
            load_json(score_dir/'generated_scores.jsonl'),
            key_col_name='label', val_col_name='scores'
        )
        self.scores_dict = {k:np.mean(v)for k,v in self.scores_dict.items()}
        pred_dict_score = build_dict_from_df_or_dicts(
            load_json(score_dir/'generated_predictions.jsonl'),
            key_col_name='label', val_col_name='predict'
        )
        gt_dict = build_dict_from_df_or_dicts(
            df, key_col_name='data_id', val_col_name='label11'
        )
        gt_dict = {str(k):v for k,v in gt_dict.items()}
        processed_res = postprocess_generation_res_to_lid(
            pred=pred_dict_score, gt=gt_dict, label_list=dfs.label_list,
            match_strategy='first exists', 
            lower_results=True,
        )
        self.pred_dict_score = processed_res['pred']
        self.gt_dict = processed_res['gt']
        self.label_list = dfs.label_list
        self.score_dir = score_dir
        pred_dic_rest = build_dict_from_df_or_dicts(
            load_json(path(rest_dir, 'generated_predictions.jsonl')),
            key_col_name='label', val_col_name='predict'
        )
        self.pred_dict_rest = postprocess_generation_res_to_lid(
            pred=pred_dic_rest, label_list=self.label_list,
        )['pred']
        assert all(
            sorted(cdic.keys())==sorted(gt_dict.keys())
            for cdic in [self.pred_dict_score, self.pred_dict_rest, self.scores_dict]
        )
        self.rest_dir = rest_dir
        print(self.label_list)
        self.pred_dict_score, self.pred_dict_rest, self.scores_dict, self.gt_dict
        
        # get sorted_samples
        sorted_samples = []
        for data_id in self.gt_dict:
            sorted_samples.append([
                self.scores_dict[data_id], self.pred_dict_score[data_id],
                self.pred_dict_rest[data_id], self.gt_dict[data_id],
            ])
        sorted_samples.sort()
        self.sorted_samples = sorted_samples
    
    def eval(
        self, 
        draw=True, 
        split_piece=None,
        result_dir='/home/user/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/train',
        png_name='score_acc.png', 
        xlabel='confidence score', ylabel='acc', title='train',
        test_score_dir='/home/user/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/2024-07-08-13-41-06.main_base.ckpt7000.bs1*8_lr0.0001_ep5',
        test_rest_dir='/home/user/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/2024-07-06-15-09-34.main_distill_all_thp.ckpt8000.bs1*8_lr0.0001_ep5',
        test_dfs=None, 
        test_df_split='test',
    ):
        from scipy.ndimage import gaussian_filter1d
        result_dir = path(result_dir)
        
        output_string = []
        threshold_lst = []  
        fig, ax_list = plt.subplots(nrows=4,ncols=1,figsize=(8,4.8*4))
        plt.subplots_adjust(hspace=0.5)
        for lid, label in enumerate(self.label_list):
            score_lst, acc_lst = self.cal_target_acc(lid)
            score_lst, acc_lst = self.postprocess_metric(score_lst, acc_lst, split_piece=split_piece)
            threshold_pid = self.get_threshold(score_lst, acc_lst)
            score_threshold = score_lst[threshold_pid]
            threshold_lst.append(score_threshold)
            output_string.append(f'{label}: {score_threshold:.2f}')

            # plot
            cur_ax = ax_list[lid]
            cur_ax.plot(score_lst, acc_lst, 
                        label=str(split_piece),
                        )
            cur_ax.plot(score_lst[threshold_pid], acc_lst[threshold_pid],
                        marker='o')
            cur_ax.set_xlabel(xlabel)
            cur_ax.set_ylabel(ylabel)
            cur_ax.set_title(f'{label}')
            if split_piece:
                cur_ax.legend()
        output_string.append(', '.join(map(str, threshold_lst)))
        output_string = '\n'.join(output_string)
        print(output_string)
        make_path(result_dir)
        with open(result_dir/'thresholds.txt', 'w', encoding='utf8')as f:
            f.write(output_string)
    
        final_res = calculate_metric(
            score_dir=test_score_dir,
            rest_dir=test_rest_dir,
            dfs=test_dfs if test_dfs else self.dfs,
            split=test_df_split,
            confidence_score_threshold=threshold_lst+[100000],
        )
        final_res = json.dumps(final_res, indent=4, ensure_ascii=False)
        # print(final_res)
        with open(result_dir/'final_res.json', 'w', encoding='utf8')as f:
            f.write(final_res)
                
        if title:
            plt.title(title)
        if draw:
            plt.savefig(result_dir/png_name)
        
        return threshold_lst
    
    def cal_target_f1(
        self,
        # score_pred_dict, rest_pred_dict, gt_dict, 
        # score_dict, 
        target, target_threshold,
    ):
        raise 'TODO'
        score_pred_dict = self.pred_dict_score
        rest_pred_dict = self.pred_dict_rest
        score_dict = self.scores_dict
        gt_dict = self.gt_dict
        
        pred_lst, gt_lst = [], []
        for did in gt_dict:
            gt_lst.append(gt_dict[did] == target)
            cur_score_pred = score_pred_dict[did]
            cur_rest_pred = rest_pred_dict[did]
            cur_score = score_dict[did]
            if cur_score_pred==target:
                if cur_score>target_threshold:
                    pred_lst.append(True)
                else:
                    pred_lst.append(cur_rest_pred == target)
            else:
                pred_lst.append(False)
        cls_report = classification_report(
            y_true=gt_lst, y_pred=pred_lst, output_dict=True,
        )
        # print(classification_report(
        #     y_true=gt_lst, y_pred=pred_lst, output_dict=False,
        # ))
        metrics_dict = {
            'f1':cls_report['macro avg']['f1-score'],
            'cls_report': cls_report,
        }
        return metrics_dict
    
    def cal_target_acc(self, target):
        score_lst, acc_lst = [], []
        acc_cnt = 0
        acc_init = 0
        for score, score_pred, rest_pred, gt in self.sorted_samples:
            if score_pred != target:
                continue
            score_lst.append(score)
            if score_pred == gt:
                acc_cnt -= 1
                acc_init += 1
            if rest_pred == gt:
                acc_cnt += 1
            acc_lst.append(acc_cnt)
        acc_lst = [(p+acc_init)/len(acc_lst) for p in acc_lst]
        return score_lst, acc_lst

    def postprocess_metric(self, score_lst, metric_lst, split_piece=None):
        from scipy.ndimage import gaussian_filter1d
        if split_piece:
            nx, ny = [], []
            cnt = len(score_lst)
            gap = cnt//split_piece
            for p in range(0, cnt, gap):
                nx.append(np.mean(score_lst[p:p+gap]))
                ny.append(np.mean(metric_lst[p:p+gap]))
            score_lst, metric_lst = nx, ny
        metric_lst = gaussian_filter1d(metric_lst, sigma=2)
        return score_lst, metric_lst
  
    def get_threshold(self, score_lst, metric_lst):
        pid = max(
            range(len(score_lst)),
            key=lambda x: (metric_lst[x], -score_lst[x]),
        )
        return pid
        
      
if __name__ == '__main__':
    score_dir = '/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb3-top/2024-07-12-14-13-26.main_base_train.ckpt7000.bs1*8_lr0.0001_ep5'
    rest_dir = '/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb3-top/2024-07-12-14-14-31.main_distill_all_thp_train.ckpt8000.bs1*8_lr0.0001_ep5'
    test_score_dir='/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb3-top/2024-07-08-13-41-06.main_base.ckpt7000.bs1*8_lr0.0001_ep5 copy',
    test_rest_dir='/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb3-top/2024-07-06-15-09-34.main_distill_all_thp.ckpt8000.bs1*8_lr0.0001_ep5',
    df_split = 'train'
    split_piece = None
    result_dir = path(
        '/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb3-top',
        f'{df_split}_{split_piece}'
    )
    
    score_evalor = ConfidenceScoresEvaluator(
        dfs=IDRRDataFrames(
            data_name='pdtb3', data_level='top', data_relation='Implicit',
            data_path='/home/user/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv'
            # data_path='/home/user/test/zpwang/IDRR_data/data/used/pdtb3.p1.csv'
        ),
        split=df_split,
        score_dir=score_dir,
        rest_dir=rest_dir,
    )
    score_evalor.eval(
        draw=True, 
        split_piece=split_piece,
        result_dir=result_dir,
        png_name=f'{df_split}_score_acc.png',
        xlabel='confidence score', ylabel='acc',
        test_score_dir=test_score_dir,
        test_rest_dir=test_rest_dir,
        title=df_split,
        test_dfs=None,
        test_df_split='test',
    )
    