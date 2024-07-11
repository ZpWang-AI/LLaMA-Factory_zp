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
        rest_dir,
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
        self.label_list = dfs.label_list
        self.target_res_dir = target_res_dir
        rest_pred = build_dict_from_df_or_dicts(
            load_json(path(rest_dir, 'generated_predictions.jsonl')),
            key_col_name='label', val_col_name='predict'
        )
        self.rest_pred = postprocess_generation_res_to_lid(
            pred=rest_pred, label_list=self.label_list,
        )['pred']
        assert sorted(pred_dict.keys())==sorted(gt_dict.keys())==sorted(self.scores_dict.keys())==sorted(self.rest_pred.keys())
        self.rest_dir = rest_dir
        print(processed_res['label_list'])
        # assert len(pred_dict)==len(gt_dict)==len(self.scores_dict)
  
    def draw2(self, draw_max=False):
        # if draw_max:
        #     to_draw_dots_list = self.get_thresholds2()['threshold_pids']
        # else:
        #     draw_max = None
        fig, ax_list = plt.subplots(nrows=4,ncols=1,figsize=(8,4.8*4))
        plt.subplots_adjust(hspace=0.5)
        for lid, label in enumerate(self.label_list):
            cur_ax = ax_list[lid]
            self._draw_confidence_score_acc2(
                **self._get_target_confidence_score_correctness2(target=lid),
                ax=cur_ax, 
                # to_draw_dots=[to_draw_dots_list[lid]],
            )
            cur_ax.set_title(f'{label}')
        
        # plt.gca().invert_xaxis()
        plt.savefig(self.target_res_dir/f'confidence_score2.png')
    
    def _get_target_confidence_score_correctness2(self, target):
        conf_score = []
        correctness = []
        rest_correct = []
        for data_id in self.gt_dict:
            if self.pred_dict[data_id] == target:
                correctness.append(self.pred_dict[data_id] == self.gt_dict[data_id])
                rest_correct.append(self.rest_pred[data_id] == self.gt_dict[data_id])
                conf_score.append(ConfidenceScoresEvaluator.scores_to_confidence_scores(
                        self.scores_dict[data_id]
                ))
        conf_score, correctness, rest_correct = zip(
            *sorted(zip(conf_score,correctness,rest_correct), reverse=True))
        
        acc = []
        total_acc = []
        right = 0
        total_right = sum(rest_correct)
        cnt = len(conf_score)
        
        for pid, (p,rp) in enumerate(zip(correctness, rest_correct)):
            if p:
                right += 1
                total_right += 1
            if rp:
                total_right -= 1
            acc.append(right/(pid+1))
            total_acc.append(total_right/cnt)
        return {
            'conf_score': conf_score,
            'correctness': correctness,
            'acc': acc,
            'total_acc': total_acc,
            'cnt': cnt,
        }
    
    def _draw_confidence_score_acc2(
        self, conf_score, correctness, acc, total_acc, cnt, 
        ax, to_draw_dots=None,
    ):
        from scipy.ndimage import gaussian_filter1d

        acc = gaussian_filter1d(acc, sigma=1)

        def draw_by_gap2(gap, label):
            nx,ny = [],[]
            for p in range(0,cnt,gap):
                # cx = p/cnt
                cp = p/cnt
                cx = np.mean(conf_score[p:p+gap])
                cy = np.mean(total_acc[p:p+gap])
                nx.append(cx)
                ny.append(cy)
            
            cut_len = len(nx)//10
            # nx = nx[cut_len:]
            # ny = ny[cut_len:]
            
            # plt.plot(nx,ny, marker='o')
            ny = gaussian_filter1d(ny, sigma=2)
            ax.plot(nx,ny,
                    label=str(label)
                    )
            did = max(
                range(len(nx)),
                key=lambda x: (ny[x],x)
            )
            ax.plot(nx[did],ny[did],marker='o')
            ax.set_xlabel('confidence score')
            ax.set_ylabel('dev_precision')
        # draw_by_gap2(cnt//5,5)
        # draw_by_gap2(cnt//10,10)
        # draw_by_gap2(cnt//15,15)
        # draw_by_gap2(cnt//20,20)
        # draw_by_gap2(cnt//25,25)
        draw_by_gap2(cnt//30,30)
        draw_by_gap2(cnt//50,50)
        draw_by_gap2(cnt//100,100)
                
        # ax.invert_xaxis()
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
      
    def get_thresholds2(self, show=False):
        conf_score_thresholds = []
        threshold_pids = []
        for lid, label in enumerate(self.label_list):
            ans_dict = self._get_target_confidence_score_correctness2(target=lid)
            # print(ans_dict)
            pid = max(
                range(ans_dict['cnt']), 
                key=lambda x: (ans_dict['total_acc'][x], x)
            )
            conf_score_thresholds.append(ans_dict['conf_score'][pid])
            threshold_pids.append(pid)
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
        rest_dir='/home/qwe/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/2024-07-06-15-09-34.main_distill_all_thp.ckpt8000.bs1*8_lr0.0001_ep5',
    )
    # score_evalor.get_thresholds2(show=True)
    score_evalor.draw2(draw_max=True)