from utils_zp.common_import import *

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support, 
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
    cls_report = classification_report(
        y_true=gt, y_pred=pred, 
        labels=list(range(len(label_list))), 
        target_names=label_list, zero_division=0,
        output_dict=True,
    )
    res_dic = {
        'macro-f1': cls_report['macro avg']['f1-score'],
        'res_dic': json.dumps(cls_report, ensure_ascii=False, indent=4)
    }
    dump_json(res_dic, target_dir/'processed_dic.json', indent=4)


if __name__ == '__main__':
    calculate_metric('/home/qwe/test/zpwang/LLaMA/exp_space/base/2024-06-14-10-32-07.base_pred.ckpt4000.bs1*8_lr0.0001_ep5')