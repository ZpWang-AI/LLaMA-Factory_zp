from utils_zp import *
from IDRR_data import IDRRDataFrames

import numpy as np
from sklearn.metrics import classification_report


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
        self.scores_dict = {k:np.mean(v)for k,v in self.scores_dict.items()}
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
        self.rest_pred_dict = postprocess_generation_res_to_lid(
            pred=rest_pred, label_list=self.label_list,
        )['pred']
        assert sorted(pred_dict.keys())==sorted(gt_dict.keys())==sorted(self.scores_dict.keys())==sorted(self.rest_pred_dict.keys())
        self.rest_dir = rest_dir
        print(processed_res['label_list'])
        # assert len(pred_dict)==len(gt_dict)==len(self.scores_dict)
        
        self._conf_metric_lst = []
        for lid, label in enumerate(self.label_list):
            pass
  
    def draw3(self, tar_png='confidence_score2.png'):
        fig, ax_list = plt.subplots(nrows=4,ncols=1,figsize=(8,4.8*4))
        plt.subplots_adjust(hspace=0.5)
        for lid, label in enumerate(self.label_list):
            cur_ax = ax_list[lid]
            self._draw_confidence_score_acc3(
                **self._get_target_confidence_score_correctness3(target=lid),
                ax=cur_ax, 
                # to_draw_dots=[to_draw_dots_list[lid]],
            )
            cur_ax.set_title(f'{label}')
        
        # plt.gca().invert_xaxis()
        plt.savefig(self.target_res_dir/tar_png)
    
    # @staticmethod
    def cal_target_confidence_score_metrics(
        self,
        # score_pred_dict, rest_pred_dict, gt_dict, 
        # score_dict, 
        target, target_threshold,
    ):
        score_pred_dict = self.pred_dict
        rest_pred_dict = self.rest_pred_dict
        gt_dict = self.gt_dict
        score_dict = self.scores_dict
        pred_lst, gt_lst = [], []
        for did in gt_dict:
            gt_lst.append(gt_dict[did]==target)
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
    
    def _get_target_confidence_score_correctness3(self, target):
        
        score_lst = sorted(self.scores_dict.values())
        f1_lst = [self.cal_target_confidence_score_metrics(
            target=target, target_threshold=score
        )['f1'] for score in score_lst]
        # print(f1_lst)
        return {
            'f1': f1_lst,
            'conf_score': score_lst,
            'cnt': len(f1_lst),
        }
        return {
            'conf_score': conf_score,
            'correctness': correctness,
            'acc': acc,
            'total_acc': total_acc,
            'cnt': cnt,
        }
    
    def _draw_confidence_score_acc3(
        self, conf_score=None, correctness=None, 
        acc=None, total_acc=None, cnt=None, 
        ax=None, to_draw_dots=None,
        f1=None, score=None,
    ):
        # print(f1)
        from scipy.ndimage import gaussian_filter1d

        # acc = gaussian_filter1d(acc, sigma=1)

        def draw_by_gap2(gap, label):
            nx,ny = [],[]
            for p in range(0,cnt,gap):
                # cx = p/cnt
                cp = p/cnt
                cx = np.mean(conf_score[p:p+gap])
                cy = np.mean(f1[p:p+gap])
                nx.append(cx)
                ny.append(cy)
            
            cut_len = len(nx)//10
            # nx = nx[cut_len:]
            # ny = ny[cut_len:]
            
            # plt.plot(nx,ny, marker='o')
            # ny = gaussian_filter1d(ny, sigma=2)
            ax.plot(nx,ny,
                    label=str(label)
                    )
            did = max(
                range(len(nx)),
                key=lambda x: (ny[x],-x)
            )
            ax.plot(nx[did],ny[did],marker='o')
            ax.set_xlabel('confidence score')
            ax.set_ylabel('train_precision')
        # draw_by_gap2(cnt//5,5)
        # draw_by_gap2(cnt//10,10)
        # draw_by_gap2(cnt//15,15)
        # draw_by_gap2(cnt//20,20)
        # draw_by_gap2(cnt//25,25)

        # for tp in [10,30,50,100,150]:
        # for tp in [200,250,300,400,500]:
        # for tp in [5000]:
        #     draw_by_gap2(cnt//tp, tp)
        draw_by_gap2(1,'all')
        # draw_by_gap2(cnt//30,30)
        # draw_by_gap2(cnt//50,50)
        # draw_by_gap2(cnt//100,100)
        # draw_by_gap2(cnt//150,150)
        # draw_by_gap2(cnt//200,200)
        # draw_by_gap2(cnt//250,250)
                
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
      
    def get_thresholds3(self, show=False):
        conf_score_thresholds = []
        threshold_pids = []
        for lid, label in enumerate(self.label_list):
            ans_dict = self._get_target_confidence_score_correctness3(target=lid)
            # print(ans_dict)
            pid = max(
                range(ans_dict['cnt']), 
                key=lambda x: (ans_dict['f1'][x], x)
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
        split='train',
        # target_res_dir='/home/qwe/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/2024-07-12-14-13-26.main_base_train.ckpt7000.bs1*8_lr0.0001_ep5',
        # rest_dir='/home/qwe/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/2024-07-12-14-14-31.main_distill_all_thp_train.ckpt8000.bs1*8_lr0.0001_ep5',
        target_res_dir='/home/qwe/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/2024-07-12-14-13-26.main_base_train.ckpt7000.bs1*8_lr0.0001_ep5',
        rest_dir='/home/qwe/test/zpwang/LLaMA/exp_space/Main_distill_all_confidence/2024-07-12-14-14-31.main_distill_all_thp_train.ckpt8000.bs1*8_lr0.0001_ep5'
    )
    # print(score_evalor.cal_target_confidence_score_metrics(
    #     target=0, target_threshold=30
    # ))
    score_evalor.get_thresholds3(show=True)
    # score_evalor.draw3(tar_png='score_all_f1.png')