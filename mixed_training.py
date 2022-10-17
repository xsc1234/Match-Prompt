import json
import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
import random
from os.path import join, abspath, dirname


from data_utils.vocab import init_vocab
from p_tuning.modeling_p_tunning_all_layer import PTune_bert_p_tunning_all_layer
import sys
#from torch import nn
import joblib
SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large',
                  'megatron_11b']


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0
    max_steps = 0
    max_arrow = 50
    infoDone = 'done'
    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="P1001")
    parser.add_argument("--model_name", type=str, default='bert-base-cased')
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--mid", type=int, default=0)
    parser.add_argument("--template", type=str, default="(6, 6, 0,0,0)")
    parser.add_argument("--early_stop", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=True)

    parser.add_argument("--vocab_strategy", type=str, default="shared", choices=['original', 'shared', 'lama'])
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), './data/LAMA'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), './out/'))
    parser.add_argument("--qa_train_data", type=str)
    parser.add_argument("--qa_dev_data", type=str)
    parser.add_argument("--adhoc_train_data", type=str)
    parser.add_argument("--adhoc_dev_data", type=str)
    parser.add_argument("--nli_train_data", type=str)
    parser.add_argument("--nli_dev_data", type=str)
    parser.add_argument("--pi_train_data", type=str)
    parser.add_argument("--pi_dev_data", type=str)
    parser.add_argument("--quora_train_data", type=str)
    parser.add_argument("--quora_dev_data", type=str)

    parser.add_argument("--adhoc_pt", type=str)
    parser.add_argument("--qa_pt", type=str)
    parser.add_argument("--nli_pt", type=str)
    parser.add_argument("--pi_pt", type=str)
    parser.add_argument("--dia_pt", type=str)
    # MegatronLM 11B
    parser.add_argument("--checkpoint_dir", type=str, default=join(abspath(dirname(__file__)), '../checkpoints'))

    args = parser.parse_args()

    # post-parsing args

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if self.args.model_name != 't5-11b' else 'cuda:{}'.format(self.args.t5_shard * 4)
        if self.args.use_original_template and (not self.args.use_lm_finetune) and (not self.args.only_evaluate):
            raise RuntimeError("""If use args.use_original_template is True, 
            either args.use_lm_finetune or args.only_evaluate should be True.""")

        # load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model_name else self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        init_vocab(args)


        #########load nq########
        from data_utils.dataset_ptunning import load_mli,load_csv,load_json_trivia,load_adhoc_mq_train,load_dailydialogue,load_msrp,MultiDataset_yes_no
        q_qa = 'Do these two sentences match?5'
        self.train_data_squad = load_json_trivia(args.qa_train_data,q_qa)[:40000]
        self.dev_data_squad = load_json_trivia(args.qa_dev_data,q_qa)[:5000]

        q_nli = 'Do these two sentences match?3'
        self.train_data_mli = load_mli(args.mli_train_data, q_nli)[:40000]
        self.dev_data_mli = load_mli(args.mli_dev_data,q_nli)[:5000]

        q_adhoc = 'Do these two sentences match?2'
        self.train_data_adhoc = joblib.load(args.adhoc_train_data)
        self.dev_data_adhoc = joblib.load(args.adhoc_dev_data)[:5000]
        for i in self.train_data_adhoc:
            i['q'] = q_adhoc
        for i in self.dev_data_adhoc:
            i['q'] = q_adhoc


        q_dia = 'Do these two sentences match?1'
        self.train_data_dia = joblib.load(args.dia_train_data)
        self.dev_data_dia = joblib.load(args.dia_train_data)[:5000]
        for i in self.train_data_dia:
            i['q'] = q_dia
        for i in self.dev_data_dia:
            i['q'] = q_dia

        q_pi = 'Do these two sentences match?4'

        self.train_data_pi = load_msrp(args.pi_train_data,q_pi)
        self.dev_data_pi = load_msrp(args.pi_dev_data,q_pi)

        self.train_data_quora = load_csv(args.quora_train_data, q_pi)[:36000]
        self.dev_data_quora = load_csv(args.quora_dev_data,q_pi)[:5000]


        self.train_data = self.train_data_squad + self.train_data_mli + self.train_data_adhoc + self.train_data_dia + self.train_data_pi + self.train_data_quora
        self.dev_data = self.dev_data_squad + self.dev_data_mli + self.dev_data_adhoc + self.dev_data_dia + self.dev_data_pi + self.dev_data_quora

        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)

        self.train_set = MultiDataset_yes_no('train', self.train_data)
        self.dev_set = MultiDataset_yes_no('dev', self.dev_data)
        os.makedirs(self.get_save_path(), exist_ok=True)

        self.train_loader = DataLoader(self.train_set, batch_size=15,shuffle=False,drop_last=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=16)
        q = 'The relationship between the two sentences is '
        self.model = PTune_bert_p_tunning_all_layer(args, self.device, self.args.template,q,dic_p_tunning)

    def eval_qa(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader_qa
            dataset = self.test_data_squad
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss, pos_num_epoches, pos_num_all_epoches, pre_pos_num_epoches = 0, 0, 0, 0, 0
            # tqdm_iterator = tqdm(enumerate(loader)))
            print(len(loader))
            for idx, data in tqdm(enumerate(loader)):
                # print(idx)
                x_hs = data[0]
                x_ts = data[1]
                if False and self.args.extend_data:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model.test_extend_data(
                        x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                else:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                hit1 += _hit1
                loss += _loss.item()
                pos_num_epoches += pos_num
                pos_num_all_epoches += pos_num_all
                pre_pos_num_epoches += pre_pos_num

            hit1 /= len(dataset)
            recall = pos_num_epoches / (pos_num_all_epoches + 0.1)
            pression = pos_num_epoches / (pre_pos_num_epoches + 0.1)
            f1_score = 2 * pression * recall / (pression + recall + 0.1)
            print(
                "QA: Epoch {} Loss: {} Hit@1: {} recall:{} pression:{} F1-score:{} ".format(epoch_idx,
                                                                                        loss / len(dataset),
                                                                                        hit1, recall,
                                                                                        pression, f1_score))

        return loss, f1_score, recall, pression, hit1

    def eval_nli(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader_nli
            dataset = self.test_data_mli
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss, pos_num_epoches, pos_num_all_epoches, pre_pos_num_epoches = 0, 0, 0, 0, 0
            # tqdm_iterator = tqdm(enumerate(loader)))
            print(len(loader))
            for idx, data in tqdm(enumerate(loader)):
                # print(idx)
                x_hs = data[0]
                x_ts = data[1]
                if False and self.args.extend_data:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model.test_extend_data(
                        x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                else:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                hit1 += _hit1
                loss += _loss.item()
                pos_num_epoches += pos_num
                pos_num_all_epoches += pos_num_all
                pre_pos_num_epoches += pre_pos_num

            hit1 /= len(dataset)
            recall = pos_num_epoches / (pos_num_all_epoches + 0.1)
            pression = pos_num_epoches / (pre_pos_num_epoches + 0.1)
            f1_score = 2 * pression * recall / (pression + recall + 0.1)
            print(
                "NLI: Epoch {} Loss: {} Hit@1: {} recall:{} pression:{} F1-score:{} ".format(epoch_idx,
                                                                                        loss / len(dataset),
                                                                                        hit1, recall,
                                                                                        pression, f1_score))

        return loss, f1_score, recall, pression, hit1

    def eval_adhoc(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader_adhoc
            dataset = self.test_data_adhoc
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss, pos_num_epoches, pos_num_all_epoches, pre_pos_num_epoches = 0, 0, 0, 0, 0
            # tqdm_iterator = tqdm(enumerate(loader)))
            print(len(loader))
            for idx, data in tqdm(enumerate(loader)):
                # print(idx)
                x_hs = data[0]
                x_ts = data[1]
                if False and self.args.extend_data:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model.test_extend_data(
                        x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                else:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                hit1 += _hit1
                loss += _loss.item()
                pos_num_epoches += pos_num
                pos_num_all_epoches += pos_num_all
                pre_pos_num_epoches += pre_pos_num

            hit1 /= len(dataset)
            recall = pos_num_epoches / (pos_num_all_epoches + 0.1)
            pression = pos_num_epoches / (pre_pos_num_epoches + 0.1)
            f1_score = 2 * pression * recall / (pression + recall + 0.1)
            print(
                "Adhoc: Epoch {} Loss: {} Hit@1: {} recall:{} pression:{} F1-score:{} ".format(epoch_idx,
                                                                                        loss / len(dataset),
                                                                                        hit1, recall,
                                                                                        pression, f1_score))

        return loss, f1_score, recall, pression, hit1

    def eval_pi(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader_pi
            dataset = self.test_data_pi
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss, pos_num_epoches, pos_num_all_epoches, pre_pos_num_epoches = 0, 0, 0, 0, 0
            # tqdm_iterator = tqdm(enumerate(loader)))
            print(len(loader))
            for idx, data in tqdm(enumerate(loader)):
                # print(idx)
                x_hs = data[0]
                x_ts = data[1]
                if False and self.args.extend_data:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model.test_extend_data(
                        x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                else:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                hit1 += _hit1
                loss += _loss.item()
                pos_num_epoches += pos_num
                pos_num_all_epoches += pos_num_all
                pre_pos_num_epoches += pre_pos_num

            hit1 /= len(dataset)
            recall = pos_num_epoches / (pos_num_all_epoches + 0.1)
            pression = pos_num_epoches / (pre_pos_num_epoches + 0.1)
            f1_score = 2 * pression * recall / (pression + recall + 0.1)
            print(
                "PI: Epoch {} Loss: {} Hit@1: {} recall:{} pression:{} F1-score:{} ".format(epoch_idx,
                                                                                        loss / len(dataset),
                                                                                        hit1, recall,
                                                                                        pression, f1_score))

        return loss, f1_score, recall, pression, hit1

    def eval_dia(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader_dia
            dataset = self.test_data_dia
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss, pos_num_epoches, pos_num_all_epoches, pre_pos_num_epoches = 0, 0, 0, 0, 0
            # tqdm_iterator = tqdm(enumerate(loader)))
            print(len(loader))
            for idx, data in tqdm(enumerate(loader)):
                # print(idx)
                x_hs = data[0]
                x_ts = data[1]
                if False and self.args.extend_data:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model.test_extend_data(
                        x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                else:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num, y_label, y_pre = self.model(x_hs, x_ts)
                hit1 += _hit1
                loss += _loss.item()
                pos_num_epoches += pos_num
                pos_num_all_epoches += pos_num_all
                pre_pos_num_epoches += pre_pos_num

            hit1 /= len(dataset)
            recall = pos_num_epoches / (pos_num_all_epoches + 0.1)
            pression = pos_num_epoches / (pre_pos_num_epoches + 0.1)
            f1_score = 2 * pression * recall / (pression + recall + 0.1)
            print(
                "DIA: Epoch {} Loss: {} Hit@1: {} recall:{} pression:{} F1-score:{} ".format(epoch_idx,
                                                                                        loss / len(dataset),
                                                                                        hit1, recall,
                                                                                        pression, f1_score))

        return loss, f1_score, recall, pression, hit1

    def evaluate(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
            dataset = self.test_set
        else:
            loader = self.dev_loader
            dataset = self.dev_set
        with torch.no_grad():
            self.model.eval()
            hit1, loss, pos_num_epoches, pos_num_all_epoches, pre_pos_num_epoches = 0, 0, 0, 0, 0
            # tqdm_iterator = tqdm(enumerate(loader)))
            print(len(loader))
            for idx, data in tqdm(enumerate(loader)):
                # print(idx)
                x_hs = data[0]
                x_ts = data[1]
                if False and self.args.extend_data:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num,y_label,y_pre = self.model.test_extend_data(x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num,y_label,y_pre = self.model(x_hs, x_ts)
                else:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num,y_label,y_pre = self.model(x_hs, x_ts)
                hit1 += _hit1
                loss += _loss.item()
                pos_num_epoches += pos_num
                pos_num_all_epoches += pos_num_all
                pre_pos_num_epoches += pre_pos_num

            hit1 /= len(dataset)
            recall = pos_num_epoches / (pos_num_all_epoches+0.1)
            pression = pos_num_epoches / (pre_pos_num_epoches+0.1)
            f1_score = 2 * pression * recall / (pression + recall+0.1)
            print(
                "Epoch {} Loss: {} Hit@1: {} recall:{} pression:{} F1-score:{} ".format(epoch_idx, loss / len(dataset),
                                                                                        hit1, recall,
                                                                                        pression, f1_score))

        return loss, f1_score, recall, pression, hit1

    def get_task_name(self):
        if self.args.only_evaluate:
            return "_".join([self.args.model_name + ('_' + self.args.vocab_strategy), 'only_evaluate'])
        names = [self.args.model_name + ('_' + self.args.vocab_strategy),
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search', self.get_task_name(),
                    self.args.relation_id)

    def get_checkpoint(self, epoch_idx, dev_f1s, dev_acc):
        ckpt_name = "epoch_{}_dev_f1{}_dev_acc{}.ckpt".format(epoch_idx, round(dev_f1s * 100, 4),
                                                              round(dev_acc * 100, 4))
        return {'embedding': self.model.state_dict(),
                'dev_f1s': dev_f1s,
                'dev_acc': dev_acc,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)
        for epoch_idx in range(100):
            if epoch_idx > -1:
                if epoch_idx == 0:
                    test_loss, test_f1s, test_recall, test_pression, test_hit1 = self.evaluate(epoch_idx, 'Dev')
                    best_dev = 0
                else:
                    print('Evaluating:..........................')
                    test_loss, test_f1s, test_recall, test_pression, test_hit1 = self.evaluate(epoch_idx, 'Dev')
                if epoch_idx > 0 and (test_f1s >= best_dev) or self.args.only_evaluate:
                    best_ckpt = self.get_checkpoint(epoch_idx, test_f1s, test_hit1)
                    early_stop = 0
                    best_dev = test_f1s
                    self.save(best_ckpt)
            if self.args.only_evaluate:
                break
            hit1, num_of_samples = 0, 0
            tot_loss = 0
            print(len(self.train_loader))
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                loss = self.model(batch[0], batch[1])[0]
                tot_loss += loss.item()
                num_of_samples += len(batch[0])

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
            my_lr_scheduler.step()
        self.save(best_ckpt)

        return best_ckpt

def main(relation_id=None):
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    args = construct_generation_args()
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.model_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    adhoc_p_tunning = joblib.load(args.adhoc_pt)
    nli_p_tunning = joblib.load(args.nli_pt)
    dia_p_tunning = joblib.load(args.dia_pt)
    pi_p_tunning = joblib.load(args.pi_pt)
    qa_p_tunning = joblib.load(args.qa_pt)
    # 1:'dia'
    # 2:'adhoc'
    # 3:'nli'
    # 4:'pi'
    # 5:'qa'
    dic_p_tunning = {}
    dic_p_tunning[1] = dia_p_tunning
    dic_p_tunning[2] = adhoc_p_tunning
    dic_p_tunning[3] = nli_p_tunning
    dic_p_tunning[4] = pi_p_tunning
    dic_p_tunning[5] = qa_p_tunning

    main()
