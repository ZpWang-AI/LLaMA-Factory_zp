from utils_zp.common_import import *

import pandas as pd

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support, 
    confusion_matrix
)

from utils_zp import postprocess_generation_res_to_lid, load_json, dump_json
from IDRR_data import IDRRDataFrames


def calculate_metric(target_dir, gt_dic):
    target_dir = path(target_dir)
    generated_predictions = target_dir/'generated_predictions.jsonl'
    if not generated_predictions.exists():
        return
    pred, gt = [], []
    for line in load_json(generated_predictions):
        pred.append(line['predict'])
        # gt.append(line['label'])
        gt.append(
            gt_dic[int(line['label'])]
        )
    
    postprocessed = postprocess_generation_res_to_lid(
        pred=pred, 
        gt=gt, 
        match_strategy='first exists'  # TODO
    )
    pred = postprocessed['pred']
    gt = postprocessed['gt']
    label_list = postprocessed['label_list']

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
    print_sep()
    dump_json(res_dic, target_dir/'processed_dic.json', indent=4)


if __name__ == '__main__':
    from utils_zp import build_dict_from_df_or_dicts
    gt_dic = build_dict_from_df_or_dicts(
        IDRRDataFrames(
            data_name='pdtb3',
            data_level='top', data_relation='Implicit',
            data_path='/home/user/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv',
        ).test_df,
        key_col_name='data_id', val_col_name='label11',
        make_key_str=False,
    )
    root_dir = '/home/user/test/zpwang/LLaMA/exp_space/Main_llama_init'
    for dir in os.listdir(root_dir):
        dir = path(root_dir)/dir
        calculate_metric(dir, gt_dic)
    # calculate_metric('/home/user/test/zpwang/LLaMA/exp_space/filter/2024-06-21-20-37-57.filter.ckpt2000.bs1*8_lr0.0001_ep5')