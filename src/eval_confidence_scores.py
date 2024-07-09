from utils_zp import *
from IDRR_data import IDRRDataFrames

import numpy as np


class ConfidenceScoresEvaluator:
    @staticmethod
    def scores_to_confidence_scores(scores):
        return np.mean(scores)
        
    def __init__(
        self, 
        dfs:IDRRDataFrames, split, 
        target_res_dir,
    ) -> None:
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
        self.label_list = dfs.label_list
        self.target_res_dir = target_res_dir
        print(processed_res['label_list'])
        # assert len(pred_dict)==len(gt_dict)==len(self.scores_dict)
    
    def draw(self, target_acc_list=None):
        if target_acc_list:
            thresholds_res = self.get_thresholds(target_acc_list, show=False)
            conf_score_thresholds = thresholds_res['conf_score_thresholds']
            threshold_pids = thresholds_res['threshold_pids']
        
        fig, ax_list = plt.subplots(nrows=4,ncols=1,figsize=(8,4.8*4))
        plt.subplots_adjust(hspace=0.5)
        for lid, label in enumerate(self.label_list):
            cur_ax = ax_list[lid]
            cnt = self._draw_confidence_score_acc(
                **self._get_target_confidence_score_correctness(target=lid),
                ax=cur_ax,
            )
            cur_ax.set_title(f'{label}')
            
            tx, ty = threshold_pids[lid], target_acc_list[lid]
            tx = tx/cnt
            # cur_ax.plot(tx, ty, marker='o')
            # cur_ax.text(tx, ty, str(conf_score_thresholds[lid]))
            # cur_ax.axvline(tx, ymax=ty/cur_ax.get_ylim()[1], ymin=0)
            # cur_ax.axhline(ty, xmax=tx/cur_ax.get_xlim()[1], xmin=0)
        plt.savefig(self.target_res_dir/f'confidence_score.png')

    def _get_target_confidence_score_correctness(self, target):
        conf_score = []
        correctness = []
        for data_id in self.gt_dict:
            if self.pred_dict[data_id] == target:
                correctness.append(self.pred_dict[data_id] == self.gt_dict[data_id])
                conf_score.append(ConfidenceScoresEvaluator.scores_to_confidence_scores(
                        self.scores_dict[data_id]
                ))
        conf_score, correctness = zip(*sorted(zip(conf_score,correctness), reverse=True))
        
        acc = []
        right = 0
        for pid, p in enumerate(correctness):
            if p:
                right += 1
            acc.append(right/(pid+1))
        return {
            'conf_score': conf_score,
            'correctness': correctness,
            'acc': acc,
        }
    
    def _draw_confidence_score_acc(self, conf_score, correctness, acc, ax):
        from scipy.ndimage import gaussian_filter1d

        cnt = len(acc)
        print(cnt)
        acc = gaussian_filter1d(acc, sigma=1)
        # plt.figure(figsize=(8, 4.8))
        
        # x = conf_score
        # plt.gca().invert_xaxis()
        # x = range(cnt)
        
        # plt.plot(conf_score, acc, marker='o', linestyle='-')
        # plt.plot(range(len(acc),0,-1),acc)
        # plt.plot(x,acc, )
        def draw_by_gap(gap, label):
            nx,ny = [],[]
            for p in range(0,cnt,gap):
                cx = p/cnt
                cy = np.mean(acc[p:p+gap])
                nx.append(cx)
                ny.append(cy)
            
            cut_len = len(nx)//10
            nx = nx[cut_len:]
            ny = ny[cut_len:]
            
            # plt.plot(nx,ny, marker='o')
            ax.plot(nx,ny,
                    label=str(label)
                    )
            ax.set_xlabel('sample_ratio')
            ax.set_ylabel('acc')

        draw_by_gap(cnt//5,5)
        draw_by_gap(cnt//8,8)
        draw_by_gap(cnt//10,10)
        draw_by_gap(cnt//15,15)
        draw_by_gap(cnt//30,30)
        draw_by_gap(cnt//50,50)
        ax.legend()
        # gap = 15
            # plt.text(cx,cy, f'{conf_score[p]:.2f}', 
            #          horizontalalignment='center', 
            #          verticalalignment='bottom')
            # plt.plot(cx,cy, marker='o')
        # ny = gaussian_filter1d(ny, sigma=5)
            # print(p, acc[p], conf_score[cnt-1-p])
        # plt.legend()
        return cnt
        # plt.savefig(png_path)
        # plt.close()
    
    def get_thresholds(self, target_acc_list, show=False):
        conf_score_thresholds = []
        threshold_pids = []
        for lid, label in enumerate(self.label_list):
            ans_dict = self._get_target_confidence_score_correctness(target=lid)
            # print(ans_dict)
            tar_acc = target_acc_list[lid]
            for pid, cur_acc, cur_conf_score in zip(range(len(ans_dict['acc'])-1,-1,-1), ans_dict['acc'][::-1], ans_dict['conf_score'][::-1]):
                if cur_acc > tar_acc:
                    conf_score_thresholds.append(cur_conf_score)
                    threshold_pids.append(pid)
                    break
        if show:
            for cst, label in zip(conf_score_thresholds, self.label_list):
                print(label)
                print(f'{label}: {cst:.2f}')
            print(', '.join(map(lambda s:f'{s:.2f}', conf_score_thresholds)))
        return {
            'conf_score_thresholds': conf_score_thresholds,
            'threshold_pids': threshold_pids,
        }
    
    
if __name__ == '__main__':
    score_evalor = ConfidenceScoresEvaluator(
        dfs=IDRRDataFrames(
            data_name='pdtb3', data_level='top', data_relation='Implicit',
            data_path='/home/qwe/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv'
            # data_path='/home/qwe/test/zpwang/IDRR_data/data/used/pdtb3.p1.csv'
        ),
        split='test',
        target_res_dir='/home/qwe/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/2024-07-08-13-41-06.main_base.ckpt7000.bs1*8_lr0.0001_ep5 copy',
    )
    # score_evalor.draw()
    # score_evalor.get_thresholds([0.9,0.9,0.88,0.9])
    target_acc_list = [0.9,0.9,0.88,0.9]
    # target_acc_list = [0.9]*4
    score_evalor.get_thresholds(target_acc_list, show=True)
    # score_evalor.draw(target_acc_list)
    