from utils_zp.common_import import *

import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support, 
    confusion_matrix
)

from utils_zp import postprocess_generation_res_to_lid, load_json, dump_json, build_dict_from_df_or_dicts
from IDRR_data import IDRRDataFrames


def calculate_metric(
    score_dir, rest_dir, 
    dfs:IDRRDataFrames, split, 
    confidence_score_threshold: Union[List[float], Dict[int, float]],
):
    score_dir = path(score_dir)
    rest_dir = path(rest_dir)
    df = dfs.get_dataframe(split)
    scores_dict = build_dict_from_df_or_dicts(
        load_json(score_dir/'generated_scores.jsonl'),
        key_col_name='label', val_col_name='scores'
    )
    pred_dict_score = build_dict_from_df_or_dicts(
        load_json(score_dir/'generated_predictions.jsonl'),
        key_col_name='label', val_col_name='predict'
    )
    pred_dict_rest = build_dict_from_df_or_dicts(
        load_json(rest_dir/'generated_predictions.jsonl'),
        key_col_name='label', val_col_name='predict'
    )
    gt_dict = build_dict_from_df_or_dicts(
        df, key_col_name='data_id', val_col_name='label11', make_key_str=True,
    )
    postprocessed_gt = postprocess_generation_res_to_lid(
        pred=pred_dict_score, gt=gt_dict, label_list=dfs.label_list, 
        match_strategy='first exists', lower_results=True,
    )
    pred_dict_score = postprocessed_gt['pred']
    gt_dict = postprocessed_gt['gt']
    pred_dict_rest = postprocess_generation_res_to_lid(
        pred=pred_dict_rest, label_list=dfs.label_list, 
        match_strategy='first exists', lower_results=True,
    )['pred']
    label_list = dfs.label_list
    assert all(
        sorted(cdic.keys())==sorted(gt_dict.keys())
        for cdic in [pred_dict_score, pred_dict_rest, scores_dict]
    )
    
    gt, pred = [], []
    for data_id in gt_dict:
        gt.append(gt_dict[data_id])
        if np.mean(scores_dict[data_id]) < confidence_score_threshold[pred_dict_score[data_id]]:
            pred.append(pred_dict_rest[data_id])
        else:
            pred.append(pred_dict_score[data_id])

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
    # print(classification_report(
    #     y_true=gt, y_pred=reasoning, labels=list(range(len(label_list))),
    #     target_names=label_list, output_dict=False
    # ))
    # print_sep()
    macrof1 = cls_report['macro avg']['f1-score']
    acc = (confusion_mat*np.eye(len(confusion_mat))).sum() / confusion_mat.sum()
    macrof1 = f'{macrof1*100:.3f}'
    acc = f'{acc*100:.3f}'
    res_dic = {
        'macro-f1': macrof1,
        'acc': acc,
        'confusion_matrix': confusion_mat.tolist(),
        'cls_report': cls_report,
    }
    print({
        'macro-f1': macrof1,
        'acc': acc,
    })
    return res_dic
    # dump_json(res_dic, target_dir/'processed_dic.json', indent=4)


if __name__ == '__main__':
    calculate_metric(
        score_dir='/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb2-top/2024-08-31-10-25-30.main_base.ckpt6000',
        rest_dir='/home/user/test/zpwang/LLaMA/exp_space/paper_needed/main/pdtb2-top/2024-08-31-10-34-07.main_subtext.ckpt6000',
        dfs=IDRRDataFrames(
            data_name='pdtb2', data_level='top', data_relation='Implicit',
            # data_path='/home/user/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv'
            data_path='/home/user/test/zpwang/IDRR_data/data/dataBuild/pdtb2_top_implicit.subtext_distill.llama_distill_all.csv'
        ),
        split='test',
        confidence_score_threshold=[
            30.838817596435547, 35.21381711959839, 31.16776466369629, 30.77302837371826
        ],
    )