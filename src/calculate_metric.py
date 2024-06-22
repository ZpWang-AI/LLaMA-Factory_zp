from utils_zp.common_import import *

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support, 
    confusion_matrix
)

from utils_zp import postprocess_generation_res_to_lid, load_json, dump_json


def calculate_metric(target_dir):
    target_dir = path(target_dir)
    generated_predictions = target_dir/'generated_predictions.jsonl'
    pred, gt = [], []
    for line in load_json(generated_predictions):
        pred.append(line['predict'])
        # gt.append(line['label'])
        gt.append(line['label'].split(',')[0])
    postprocessed = postprocess_generation_res_to_lid(
        pred=pred, 
        gt=gt, 
        match_strategy='complete'  # TODO
    )
    pred = postprocessed['pred']
    gt = postprocessed['gt']
    label_list = postprocessed['label_list']

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
        'yes_precision': cls_report['yes']['precision'],
        'confusion_matrix': confusion_mat.tolist(),
        'cls_report': cls_report,
    }
    dump_json(res_dic, target_dir/'processed_dic.json', indent=4)


if __name__ == '__main__':
    for dir in '''
    /home/qwe/test/zpwang/LLaMA/exp_space/filter/2024-06-21-20-37-57.filter.ckpt2000.bs1*8_lr0.0001_ep5
    /home/qwe/test/zpwang/LLaMA/exp_space/filter/2024-06-21-21-57-37.filter.ckpt4000.bs1*8_lr0.0001_ep5
    /home/qwe/test/zpwang/LLaMA/exp_space/filter/2024-06-22-09-54-06.filter.ckpt6000.bs1*8_lr0.0001_ep5
    /home/qwe/test/zpwang/LLaMA/exp_space/filter/2024-06-22-09-54-36.filter.ckpt8000.bs1*8_lr0.0001_ep5
    /home/qwe/test/zpwang/LLaMA/exp_space/filter/2024-06-22-09-55-00.filter.ckptfinal.bs1*8_lr0.0001_ep5
    '''.split():
        dir = dir.strip()
        if dir:
            calculate_metric(dir)
    # calculate_metric('/home/qwe/test/zpwang/LLaMA/exp_space/filter/2024-06-21-20-37-57.filter.ckpt2000.bs1*8_lr0.0001_ep5')