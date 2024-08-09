from utils_zp.common_import import *

import pandas as pd

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support, 
    confusion_matrix
)

from utils_zp import postprocess_generation_res_to_lid, load_json, dump_json
from IDRR_data import IDRRDataFrames


def get_gt_dic(gt_df:pd.DataFrame, gt_column_name):
    return dict(zip(gt_df['data_id'], gt_df[gt_column_name]))


def calculate_metric(target_dirs, gt_dic):
    gt = []
    preds = []
    for target_dir in target_dirs:
        target_dir = path(target_dir)
        generated_predictions = target_dir/'generated_predictions.jsonl'
        if not generated_predictions.exists():
            return
        cur_pred = []
        for line in load_json(generated_predictions):
            cur_pred.append(line['predict'])
        preds.append(cur_pred)
        
        if not gt:
            if str(load_json(generated_predictions)[0]['label']).isnumeric():
                for line in load_json(generated_predictions):
                    gt.append(
                        gt_dic[int(line['label'])]
                    )
            else:
                for line in load_json(generated_predictions):
                    gt.append(
                        line['label'].split(',')[0]
                    )
    
    postprocessed_gt = postprocess_generation_res_to_lid(
        gt=gt, match_strategy='complete'
    )
    gt = postprocessed_gt['gt']
    label_list = postprocessed_gt['label_list']
    preds = [
        postprocess_generation_res_to_lid(
            pred=pred, label_list=label_list, 
            match_strategy='complete', lower_results=False,
        )['pred']
        for pred in preds
    ]
    
    pred = []
    for cur_preds, cur_gt in zip(zip(*preds), gt):
        if cur_gt in cur_preds:
            pred.append(cur_gt)
        else:
            pred.append(cur_preds[-1])

    # TODO
    confusion_mat = confusion_matrix(
        y_true=gt, y_pred=pred,
        labels=list(range(len(label_list))),
    )
    cls_report = classification_report(
        y_true=gt, y_pred=pred, 
        labels=list(range(len(label_list))), 
        target_names=label_list, zero_division=0,
        output_dict=True,
    )
    print(confusion_mat)
    print_sep()
    print(classification_report(
        y_true=gt, y_pred=pred, 
        labels=list(range(len(label_list))), 
        target_names=label_list, zero_division=0,
        output_dict=False,
    ))
    res_dic = {
        'macro-f1': cls_report['macro avg']['f1-score'],
        # 'yes_precision': cls_report['yes']['precision'],
        'confusion_matrix': confusion_mat.tolist(),
        'cls_report': cls_report,
    }
    print(res_dic)
    # dump_json(res_dic, target_dir/'processed_dic.json', indent=4)


if __name__ == '__main__':
    gt_dic = get_gt_dic(
        IDRRDataFrames(
            data_name='pdtb3',
            data_level='top', data_relation='Implicit',
            data_path='/home/user/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv',
        ).test_df,
        'label11'
    )
    calculate_metric(gt_dic=gt_dic, target_dirs=[
        '/home/user/test/zpwang/LLaMA/exp_space/Done/Main_base/2024-06-14-10-00-00.base_pred.ckptfinal.bs1*8_lr0.0001_ep5',
        '/home/user/test/zpwang/LLaMA/exp_space/Done/Main_base/2024-06-14-10-32-07.base_pred.ckpt4000.bs1*8_lr0.0001_ep5',
        # '/home/user/test/zpwang/LLaMA/exp_space/Done/Main_base/2024-06-14-10-33-17.base_pred.ckpt8000.bs1*8_lr0.0001_ep5',
        '/home/user/test/zpwang/LLaMA/exp_space/Done/Main_base/2024-06-14-11-14-18.base_pred.ckpt8000.bs1*8_lr0.0001_ep5'
        
        # '/home/user/test/zpwang/LLaMA/exp_space/Done/Main_distill_all/2024-06-24-17-22-42.main_distill_all.ckpt8000.bs1*8_lr0.0001_ep5'
    ])
    # calculate_metric('/home/user/test/zpwang/LLaMA/exp_space/filter/2024-06-21-20-37-57.filter.ckpt2000.bs1*8_lr0.0001_ep5')