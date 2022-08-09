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
from p_tuning.modeling_all_layer import PTune_bert_all_layer
import sys
#from torch import nn

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
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度
    infoDone = 'done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
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
    parser.add_argument("--model_name", type=str, default='bert-base-cased')
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

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
    parser.add_argument("--use_lm_finetune", type=bool, default=False)

    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--dev_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--out_dir", type=str)
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
        from data_utils.dataset_all_layer import load_json_trivia,MultiDataset
        self.train_data = load_json_trivia(args.train_data)[:40000]
        self.dev_data = load_json_trivia(args.dev_data)
        self.test_data = load_json_trivia(args.test_data)


        random.shuffle(self.train_data)
        random.shuffle(self.dev_data)
        random.shuffle(self.test_data)

        self.test_set = MultiDataset('test', self.test_data)
        self.train_set = MultiDataset('train', self.train_data)
        self.dev_set = MultiDataset('dev', self.dev_data)
        os.makedirs(self.get_save_path(), exist_ok=True)

        self.train_loader = DataLoader(self.train_set, batch_size=16, shuffle=True, drop_last=True)
        self.dev_loader = DataLoader(self.dev_set, batch_size=16)
        self.test_loader = DataLoader(self.test_set, batch_size=16)
        # print(self.device)
        q = 'Do these two sentences match?'
        self.model = PTune_bert_all_layer(args, self.device, self.args.template,q)


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
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num = self.model.test_extend_data(x_hs, x_ts)
                elif evaluate_type == 'Test':
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num = self.model(x_hs, x_ts)
                else:
                    _loss, _hit1, pos_num, pos_num_all, pre_pos_num = self.model(x_hs, x_ts)
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
        return {'embedding': self.model.prompt_encoder.state_dict(),
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
        params = [{'params': self.model.prompt_encoder.parameters()}]
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': 5e-6})
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)
        # for name,parameters in self.model.named_parameters():
        #     print(name)
        #     print(parameters.requires_grad)
        for epoch_idx in range(100):
            # check early stopping

            if epoch_idx > -1:
                if epoch_idx == 0:
                    #test_loss, test_f1s, test_recall, test_pression, test_hit1 = self.evaluate(epoch_idx, 'Test')
                    test_loss, test_f1s, test_recall, test_pression, test_hit1 = 0,0,0,0,0
                    best_dev = 0
                else:
                    print('Evaluating:..........................')
                    # dev_loss, dev_f1s, dev_recall, dev_pression, dev_hit1 = self.evaluate(epoch_idx, 'Dev')
                    test_loss, test_f1s, test_recall, test_pression, test_hit1 = self.evaluate(epoch_idx, 'Test')
                if epoch_idx > 0 and (test_f1s >= best_dev) or self.args.only_evaluate:
                    # test_loss, test_f1s, test_recall, test_pression, test_hit1 = self.evaluate(epoch_idx, 'Test')
                    best_ckpt = self.get_checkpoint(epoch_idx, test_f1s, test_hit1)
                    early_stop = 0
                    best_dev = test_f1s
                    self.save(best_ckpt)
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        # self.save(best_ckpt)
                        print("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
                        return best_ckpt
            if self.args.only_evaluate:
                break

            # run training
            # run training
            hit1, num_of_samples = 0, 0
            tot_loss = 0
            print(len(self.train_loader))
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                # print('batch:')
                # print(batch)
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
    args = construct_generation_args()
    if relation_id:
        args.relation_id = relation_id
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.model_name)
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    main()
