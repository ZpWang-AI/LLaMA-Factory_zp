from utils_zp import *
import_np()
import_plt()

from sklearn.metrics import f1_score, accuracy_score

    
class Analyser:
    @staticmethod
    def process_predict(fold_path:path):
        '''
        get raw output from fold_path.src_output

        get ckpt_num from last part in fold_path split by `.`
        '''
        json_path = fold_path/'src_output'/'generated_predictions.jsonl'
        if not json_path.exists():
            print(fold_path)
            print('process predict fail\n')
            return
        
        with open(fold_path/'src_output'/'all_results.json', 'r', encoding='utf8')as f:
            predict_output = json.load(f)
        predict_runtime = predict_output['predict_runtime']

        ckpt_num = str(fold_path).split('.')[-1]
        ckpt_num = re.findall(r'\d+', ckpt_num)
        if not ckpt_num:
            ckpt_num = 'final'
        else:
            ckpt_num = ckpt_num[0]
        
        labels_init, predictions_init = [], []
        with open(json_path, 'r', encoding='utf8')as f:
            for line in f.readlines():
                dic = json.loads(line)
                labels_init.append(dic['label'])
                predictions_init.append(dic['predict'])

        label_list = list(set(labels_init))
        # label_list = []
        # for label in labels_init:
        #     label_list.extend(label.split('\n'))
        # label_list = [label.strip()for label in set(label_list)if label.strip()]
        # label_list = sorted(filter(lambda x:any(x not in p for p in labels_init), label_list))
        # label_list = sorted(label.strip() for label in set(labels_init))
        # label_list =  sorted(set(labels_init))

        # def label_to_id(label_s):
        #     return label_list.index(label_s) if label_s in label_list else -1
        def label_to_id(label_s:str):
            for p, label in enumerate(label_list):
                # if label_s.startswith(label):
                if label in label_s.split('\n'):
                    return p
            return -1
        
        labels = np.array(list(map(label_to_id, labels_init))) 
        predictions = np.array(list(map(label_to_id, predictions_init)))
        
        total_num = len(labels)
        wrong_outputs = []
        for i in range(total_num):
            if predictions[i] == -1:
                wrong_outputs.append({
                    'label': labels_init[i],
                    'pred': predictions_init[i],
                    'pid': i,
                    'dir': str(fold_path),
                })
                    
        acc = (labels == predictions).mean()
        f1 = [f1_score(labels==i, predictions==i)for i in range(len(label_list))]
        macro_f1 = np.average(f1)
        
        res_dic = {
            'ckpt': ckpt_num,
            'tot': total_num,
            'acc': float(f'{acc*100:.3f}'),
            'macro-f1': float(f'{macro_f1*100:.3f}'),
            'f1': f1,
            'labels': label_list,
            'wrong': wrong_outputs,
            'pred_runtime': predict_runtime,
        }
        with open(fold_path/'result.json', 'w', encoding='utf8')as f:
            json.dump(res_dic, f, indent=2)
        return res_dic
        # result_string = '\n'.join([
        #     f'tot: {total_num}',
        #     f'acc: ',
        #     f'macro-f1: ',
        #     f'f1: {f1}',
        #     f'labels: {label_list}',
        # ])
        # # print(result_string)
        # with open(fold_path/'result.txt', 'w', encoding='utf8')as f:
        #     f.write(result_string)
        # return 
        
    @staticmethod
    def process_sft(fold_path:path):
        '''
        get raw output from fold_path.src_output
        '''
        output_path = fold_path/'src_output'/'train_results.json'
        if not output_path.exists():
            print(output_path)
            print('process sft fail')
            return
        with open(output_path, 'r', encoding='utf8')as f:
            train_output = json.load(f)
        with open(fold_path/'log', 'r', encoding='utf8')as f:
            for line in f.readlines():
                if 'Num examples' in line:
                    train_output['sample_num'] = line.split('=')[-1].strip()
                elif 'Total optimization steps' in line:
                    train_output['total_steps'] = line.split('=')[-1].strip()
                elif 'Number of trainable parameters' in line:
                    train_output['trainable_parameters'] = line.split('=')[-1].strip()
        return train_output
    
    @staticmethod
    def main_analyse(root_path):
        root_path = path(root_path)
        pred_outputs = []
        for son_fold in sorted(os.listdir(root_path)):
            if 'predict' in son_fold:
                son_fold = root_path/son_fold
                res = Analyser.process_predict(son_fold)
                pred_outputs.append(res)
            elif 'sft' in son_fold:
                son_fold = root_path/son_fold
                train_output = Analyser.process_sft(son_fold)
        
        xs, ys, pred_runtime, wrong_output = [], [], [], []
        for p in pred_outputs:
            if not p:
                continue
            if p['ckpt'] != 'final':
                xs.append(int(p['ckpt']))
            else:
                xs.append(int(train_output['total_steps'].replace(',', '')))
            ys.append(p['macro-f1'])
            pred_runtime.append(p['pred_runtime'])
            wrong_output.extend(p['wrong'])
        
        pred_runtime = np.mean(pred_runtime)
        result_dic = {
            'max macro-f1': max(ys),
            'step': str(xs),
            'macro-f1': str(ys),
            # 'train_runtime_f': format_time_seconds(train_output['train_runtime']),
            # 'pred_runtime_f': format_time_seconds(pred_runtime),
            'pred_runtime': pred_runtime, 
        } | train_output
        result_dic['wrong'] = wrong_output
        with open(root_path/'result.json', 'w', encoding='utf8')as f:
            json.dump(result_dic, f, indent=2)
        
        plt.plot(xs, ys)
        for xi, yi in zip(xs, ys):
            plt.text(xi, yi, str(yi))
        plt.xlabel('step')
        plt.ylabel('macro-f1')
        plt.title('test_macro-f1')
        img_path = root_path/'test_macro-f1'
        plt.savefig(img_path)
        plt.close()
        print(img_path.parent.name, 'saved')
        
    @staticmethod
    def cmp_analyse(target_folds, img_path, align_step=False):
        for target_fold in target_folds:
            res_json = path(target_fold)/'result.json'
            if not res_json.exists():
                Analyser.main_analyse(target_fold)
        
        if not img_path:
            return
        for target_fold in target_folds:
            res_json = path(target_fold)/'result.json'
            with open(res_json, 'r', encoding='utf8')as f:
                res_dic = json.load(f)
            xs = res_dic['step']
            if align_step:
                max_init_x = max(xs)
                xs = [int(p*align_step/max_init_x) for p in xs]
            ys = res_dic['macro-f1']
            plt.plot(xs, ys, label=str(target_fold).split('/')[-1]+f': {res_dic["max macro-f1"]}')
        
        # plt.ylim(60, 71)
        plt.legend(loc='lower center')
        plt.xlabel('step')
        plt.ylabel('macro-f1')
        plt.title('macro-f1')
        plt.savefig(img_path)
        plt.close()
        print(img_path, 'saved')
        
        
if __name__ == '__main__':
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        '--path', '-p', type=str,
        default='/public/home/hongy/zpwang/LLaMA-Factory/zpwang/experiment_space/llama27b.pdtb3.level1.NoP'
    )
    root_path = parser.parse_args().path
    Analyser.main_analyse(
        root_path=root_path
    )
    exit()