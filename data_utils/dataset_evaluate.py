from torch.utils.data import Dataset
import json
import pandas as pd
from data_utils.vocab import get_vocab_by_strategy, token_wrapper
import os
import random
import json_lines

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def load_csv(filname):
    data = []
    df = pd.read_csv(filname)
    for item in df.iterrows():
        dic_temp = {}
        dic_temp['q1'] = item[1]['question1']
        dic_temp['q2'] = item[1]['question2']
        if(item[1]['is_duplicate'] == 1):
            dic_temp['label'] = 'relevant'
        else:
            dic_temp['label'] = 'irrelevant'
        #dic_temp['label'] = item[1]['is_duplicate']
        data.append(dic_temp)
    return data

def load_msrp(filname):
    data = []
    pos = 0
    neg = 0
    with open(filname,'r') as dataset:
        lines = dataset.readlines()
        for line in lines:
            if(lines.index(line) == 0):
                continue
            item = line.split('\t')
            dic_temp = {}
            dic_temp['q1'] = item[3]
            dic_temp['q2'] = item[4][:-1]
            if(item[0] == '1'):
                dic_temp['label'] ='relevant'
                pos += 1
            else:
                dic_temp['label'] = 'irrelevant'
                neg += 1
            data.append(dic_temp)
    print(pos)
    print(neg)
    return data

def is_irrelevantrmal_word(seq):
    for word in seq:
        if(len(word) > 50 or word == '余常聞之'):
            return False
    return True
def load_json_nq(filname):
    data = []
    with open(filname, 'r') as load_f:
        dic = json.load(load_f)
        for index in range(len(dic)-1):
            for pos_idx in range(min(3,len(dic[index]['positive_ctxs']))):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                if(len(dic[index]['positive_ctxs']) == 0):
                    continue
                dic_temp['q2'] = dic[index]['positive_ctxs'][pos_idx]['text']
                dic_temp['label'] = 'relevant'
                data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            dic_temp['q2'] = dic[index+1]['positive_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            dic_temp['q2'] = dic[index]['negative_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_json_trivia(filname):
    data = []
    with open(filname, 'r') as load_f:
        dic = json.load(load_f)
        for index in range(len(dic)-1):
            for pos_idx in range(min(3,len(dic[index]['positive_ctxs']))):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                if(len(dic[index]['positive_ctxs']) == 0):
                    continue
                dic_temp['q2'] = dic[index]['positive_ctxs'][pos_idx]['text']
                dic_temp['label'] = 'relevant'
                data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            if(len(dic[index+1]['positive_ctxs']) == 0):
                continue
            dic_temp['q2'] = dic[index+1]['positive_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            if (len(dic[index]['hard_negative_ctxs']) == 0):
                continue
            dic_temp['q2'] = dic[index]['hard_negative_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_json_squad(filname):
    data = []
    with open(filname, 'r') as load_f:
        dic = json.load(load_f)
        for index in range(len(dic)-1):
            for pos_idx in range(min(3,len(dic[index]['positive_ctxs']))):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                if(len(dic[index]['positive_ctxs']) == 0):
                    continue
                dic_temp['q2'] = dic[index]['positive_ctxs'][pos_idx]['text']
                dic_temp['label'] = 'relevant'
                data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            if(len(dic[index+1]['negative_ctxs']) == 0):
                continue
            dic_temp['q2'] = dic[index]['negative_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            if (len(dic[index]['hard_negative_ctxs']) == 0):
                continue
            dic_temp['q2'] = dic[index]['hard_negative_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_adhoc(filname):
    train_data = []
    dev_data = []
    test_data = []
    dic_dataset = {}
    train_pos = 0
    train_neg = 0
    dev_pos = 0
    dev_neg = 0
    test_pos = 0
    test_neg = 0
    with open(filname + 'dataset.txt', 'r') as dataset:
        for line in dataset.readlines():
            line = line.split('\t')
            seq = line[3][:-1]
            idx = seq.find('/')
            if(not idx == -1):
                seq = seq[idx+2:]
            dic_dataset[line[1]] = seq
    dic_query = {}
    with open(filname + 'queries.txt','r') as queries:
        for line in queries.readlines():
            line = line.split('\t')
            seq = line[2][:-1]
            dic_query[line[1]] = seq
    dic_label = {}
    with open(filname + 'qrels','r') as qrels:
        for line in qrels.readlines():
            line = line.split(' ')
            q = line[0]
            d = line[2]
            label = line[3][:-1]
            if(q in dic_label.keys()):
                dic_label[q][d] = label
            else:
                dic_temp = {}
                dic_temp[d] = label
                dic_label[q] = dic_temp
    for root, dirs, files in os.walk(filname):
        # print(dirs)
        # print(root)
        # print(files)
        for file in files:
            #print(file)
            if('train' in file):
                path = root+'/'+file
                with open(path,'r') as train:
                    for line in train.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[1]]
                            label = dic_label[line[0]][line[1]]
                            if(label == '0'):
                                label = 'irrelevant'
                                train_neg += 1
                            else:
                                label = 'relevant'
                                train_pos += 1
                            dic_temp = {}
                            dic_temp['q'] = q
                            dic_temp['d'] = d
                            dic_temp['label'] = label
                            #print(dic_temp)
                            train_data.append(dic_temp)
                        except:
                            pass
            if('valid' in file):
                path = root+'/'+file
                with open(path,'r') as valid:
                    for line in valid.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[2]]
                            label = line[4]
                            if(label == '0'):
                                label = 'irrelevant'
                                dev_neg += 1
                            else:
                                label = 'relevant'
                                dev_pos += 1
                            # else:
                            #     continue
                            dic_temp = {}
                            dic_temp['q'] = q
                            dic_temp['d'] = d
                            dic_temp['label'] = label
                            dev_data.append(dic_temp)
                        except:
                            pass
            if('test' in file):
                path = root+'/'+file
                with open(path,'r') as test:
                    for line in test.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[2]]
                            label = line[4]
                            if(label == '0'):
                                label = 'irrelevant'
                                test_neg += 1
                            else:
                                label = 'relevant'
                                test_pos += 1
                            # else:
                            #     continue
                            dic_temp = {}
                            dic_temp['q'] = q
                            dic_temp['d'] = d
                            dic_temp['label'] = label
                            test_data.append(dic_temp)
                        except:
                            pass
    print(train_pos)
    print(train_neg)
    print(dev_pos)
    print(dev_neg)
    print(test_pos)
    print(test_neg)
    return train_data,dev_data,test_data

def load_adhoc_mq(filname):
    train_data = []
    dev_data = []
    test_data = []
    dic_dataset = {}
    with open(filname + 'dataset.txt', 'r') as dataset:
        for line in dataset.readlines():
            line = line.split('\t')
            seq = line[3][:-1]
            idx = seq.find('/')
            if(not idx == -1):
                seq = seq[idx+2:]
            dic_dataset[line[1]] = seq
    dic_query = {}
    with open(filname + 'queries.txt','r') as queries:
        for line in queries.readlines():
            line = line.split('\t')
            seq = line[2][:-1]
            dic_query[line[1]] = seq
    dic_label = {}
    with open(filname + 'qrels','r') as qrels:
        for line in qrels.readlines():
            line = line.split('\t')
            # print(line)
            q = line[0]
            d = line[2]
            label = line[3][:-1]
            if(q in dic_label.keys()):
                dic_label[q][d] = label
            else:
                dic_temp = {}
                dic_temp[d] = label
                dic_label[q] = dic_temp
    for root, dirs, files in os.walk(filname):
        # print(dirs)
        # print(root)
        # print(files)
        for file in files:
            #print(file)
            if('train' in file):
                path = root+'/'+file
                with open(path,'r') as train:
                    for line in train.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[1]]
                            label = dic_label[line[0]][line[1]]
                            if(label == '0'):
                                label = 'irrelevant'
                            else:
                                label = 'relevant'
                            dic_temp = {}
                            dic_temp['q'] = q
                            dic_temp['d'] = d
                            dic_temp['label'] = label
                            #print(dic_temp)
                            train_data.append(dic_temp)
                            # print(dic_temp)
                        except:
                            pass
            if('valid' in file):
                path = root+'/'+file
                with open(path,'r') as valid:
                    for line in valid.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[2]]
                            label = line[4]
                            if(label == '0'):
                                label = 'irrelevant'
                            else:
                                label = 'relevant'
                            # else:
                            #     continue
                            dic_temp = {}
                            dic_temp['q'] = q
                            dic_temp['d'] = d
                            dic_temp['label'] = label
                            dev_data.append(dic_temp)
                        except:
                            pass
            if('test' in file):
                path = root+'/'+file
                with open(path,'r') as test:
                    for line in test.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[2]]
                            label = line[4]
                            if(label == '0'):
                                label = 'irrelevant'
                            else:
                                label = 'relevant'
                            # else:
                            #     continue
                            dic_temp = {}
                            dic_temp['q'] = q
                            dic_temp['d'] = d
                            dic_temp['label'] = label
                            test_data.append(dic_temp)
                        except:
                            pass
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    return train_data[:40000],dev_data[:10000],test_data[:10000]

def load_mli(filname):
    data = []
    with open(filname,'r') as mli:
        json_data = json.load(mli)
        for item in json_data:
            dic_temp = {}
            dic_temp['q1'] = item['sentence1']
            dic_temp['q2'] = item['sentence2']
            dic_temp['label'] = item['annotator_labels']
            if(item['annotator_labels'][0] == 'entailment'):
                dic_temp['label'] = 'relevant'
            else:
                dic_temp['label'] = 'irrelevant'
            # elif(item['anirrelevanttator_labels'][0] == 'neutral'):
            #     dic_temp['label'] = 'irrelevant'
            # elif(item['anirrelevanttator_labels'][0] == 'contradiction'):
            #     dic_temp['label'] = 'Contradiction'
            data.append(dic_temp)
    return data

def load_ali(filname):
    data = []
    with open(filname,'r') as mli:
        for item in json_lines.reader(mli):
            dic_temp = {}
            dic_temp['q1'] = item['context']
            dic_temp['q2'] = item['hypothesis']
            dic_temp['label'] = item['label']
            if(item['label'] == 'e'):
                dic_temp['label'] = 'relevant'
            else:
                dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_snli(filname):
    data = []
    pos = 0
    neg = 0
    with open(filname,'r') as mli:
        json_data = json_lines.reader(mli)
        for item in json_data:
            dic_temp = {}
            dic_temp['q1'] = item['sentence1']
            dic_temp['q2'] = item['sentence2']
            dic_temp['label'] = item['annotator_labels']
            if(item['annotator_labels'][0] == 'entailment'):
                dic_temp['label'] = 'relevant'
                pos += 1
            else:
                dic_temp['label'] = 'irrelevant'
                neg += 1
            # elif(item['anirrelevanttator_labels'][0] == 'neutral'):
            #     dic_temp['label'] = 'irrelevant'
            # elif(item['anirrelevanttator_labels'][0] == 'contradiction'):
            #     dic_temp['label'] = 'Contradiction'
            data.append(dic_temp)
    print(pos)
    print(neg)
    return data

def load_scitail(filename):
    data = []
    with open(filename,'r') as scitail:
        lines = scitail.readlines()
        for line in lines:
            line = line[:-1].split('\t')
            dic_temp = {}
            dic_temp['q1'] = line[0]
            dic_temp['q2'] = line[1]
            if(line[2] == 'entails'):
                dic_temp['label'] = 'relevant'
            else:
                dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_rte(filname):
    data = []
    with open(filname,'r') as mli:
        for item in json_lines.reader(mli):
            dic_temp = {}
            dic_temp['q1'] = item['premise']
            dic_temp['q2'] = item['hypothesis']
            dic_temp['label'] = item['label']
            if(item['label'] == 'entailment'):
                dic_temp['label'] = 'relevant'
            else:
                dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_qnli(filname):
    data = []
    with open(filname,'r') as train:
        train_data = train.readlines()
        for line in train_data:
            if(train_data.index(line) == 0):
                continue
            line = line[:-1].split('\t')
            dic_tem = {}
            dic_tem['q1'] = line[1]
            dic_tem['q2'] = line[2]
            if(line[3] == 'entailment'):
                dic_tem['label'] = 'relevant'
            else:
                dic_tem['label'] = 'irrelevant'
            data.append(dic_tem)
    return data

def load_sick(filname):
    data = []
    pos = 0
    neg = 0
    with open(filname,'r') as train:
        train_data = train.readlines()
        for line in train_data:
            if(train_data.index(line) == 0):
                continue
            line = line[:-1].split('\t')
            dic_tem = {}
            dic_tem['q1'] = line[1]
            dic_tem['q2'] = line[2]
            if(line[3] == 'ENTAILMENT'):
                dic_tem['label'] = 'relevant'
                pos += 1
            else:
                dic_tem['label'] = 'irrelevant'
                neg += 1
            data.append(dic_tem)
    print(pos)
    print(neg)
    return data

def load_dialogue(filname):
    data = []
    pos = 0
    neg = 0
    with open(filname,'r') as reddit:
        lines = reddit.readlines()
        last_q2 = ''
        for line in lines:
            if('deleted' in line):
                continue
            line = line[:-1].split('__eou__')
            if(line[1] == last_q2):
                continue
            dic_temp = {}
            dic_temp['q1'] = line[0]
            dic_temp['q2'] = line[1]
            last_q2 = line[1]
            dic_temp['label'] = 'relevant'
            pos += 1
            data.append(dic_temp)
            for i in range(3): #随机采样负样本
                dic_temp = {}
                dic_temp['q1'] = line[0]
                neg_index = random.randint(0,len(lines)-1)
                neg_line = (lines[neg_index].split('__eou__'))[0]
                while neg_line == line[0]:
                    neg_index = random.randint(0, len(lines) - 1)
                    neg_line = (lines[neg_index].split('__eou__'))[0]
                neg_q2 = (lines[neg_index].split('__eou__'))[1]
                dic_temp['q2'] = neg_q2
                dic_temp['label'] = 'irrelevant'
                data.append(dic_temp)
                neg += 1
    print(pos)
    print(neg)
    return data

def load_dailydialogue(filname):
    data = []
    with open(filname,'r') as reddit:
        lines = reddit.readlines()
        last_q2 = ''
        first_line = False
        for idx in range(len(lines)-1):
            if(not '__eou__' in lines[idx+1]):
                line = lines[idx][:-1].split('__eou__')
                q1 = line[0]
                round_count = 0
                for idx_line in range(1,len(line)):
                    dic_temp = {}
                    dic_temp['q1'] = q1
                    dic_temp['q2'] = line[idx_line]
                    dic_temp['label'] = 'relevant'
                    data.append(dic_temp)
                    #print(dic_temp)
                    for i in range(3):
                        dic_temp = {}
                        dic_temp['q1'] = q1
                        neg_index = random.randint(0, len(lines) - 1)
                        if (not '__eou__' in lines[neg_index]):
                            continue
                        neg_line = (lines[neg_index].split('__eou__'))[0]
                        while neg_line == q1:
                            neg_index = random.randint(0, len(lines) - 1)
                            if (not '__eou__' in lines[neg_index]):
                                continue
                            neg_line = (lines[neg_index].split('__eou__'))[0]
                        neg_q2 = (lines[neg_index].split('__eou__'))[1]
                        dic_temp['q2'] = neg_q2
                        dic_temp['label'] = 'irrelevant'
                        data.append(dic_temp)
                        #print(dic_temp)
                    q1 = line[idx_line]
                    round_count += 1
                    if(round_count == 3):
                        break
    return data

def load_dia_movie(conversation,lines):
    data = []
    dic_line = {}
    with open(lines,'r',encoding='utf-8') as text:
        lines = text.readlines()
        for line in lines:
            line = line[:-1].split(' +++$+++ ')
            if(len(line) > 5):
                continue
            dic_line[line[0]] = line[-1]

    with open(conversation,'r',encoding='utf-8') as conver:
        lines = conver.readlines()
        for line in lines:
            try:
                line = line[:-1].split(' +++$+++ ')
                # print(line)
                con = line[-1]
                con = con[2:-2].split('\', \'')
                # print(con)
                dic_temp = {}
                dic_temp['q1'] = dic_line[con[0]]
                dic_temp['q2'] = dic_line[con[1]]
                dic_temp['label'] = 'relevant'
                data.append(dic_temp)
                #print(dic_temp)
                for i in range(3):
                    try:
                        dic_temp = {}
                        dic_temp['q1'] = dic_line[con[0]]
                        answer_index = random.randint(0,len(lines)-1)
                        answer = lines[answer_index][:-1].split(' +++$+++ ')[-1][2:-2].split('\', \'')[1]
                        dic_temp['q2'] = dic_line[answer]
                        dic_temp['label'] = 'irrelevant'
                        data.append(dic_temp)
                        #print(dic_temp)
                    except:
                        pass
            except:
                pass
                # print(dic_temp)
    random.shuffle(data)
    return data[:5000],data[5000:10000],data[10000:]

def load_bm25(filname):
    data_list = []
    with open(filname, 'r') as bm25:
        data = json.load(bm25)
        for i in range(len(data)):
            for j in range(len(data[i]['positive_ctxs']['text'])):
                dic_temp = {}
                dic_temp['q1'] = data[i]["question"]
                dic_temp['q2'] = data[i]['positive_ctxs']['text'][j][1]
                if(data[i]['positive_ctxs']['text'][j][0] > 15):
                    dic_temp['label'] = 'great'
                elif(data[i]['positive_ctxs']['text'][j][0] > 8.6):
                    dic_temp['label'] = 'high'
                elif(data[i]['positive_ctxs']['text'][j][0] > 4.32):
                    dic_temp['label'] = 'medium'
                elif(data[i]['positive_ctxs']['text'][j][0] > 1):
                    dic_temp['label'] = 'low'
                else:
                    dic_temp['label'] = 'irrelevantne'
                data_list.append(dic_temp)
                #print(dic_temp)
            for j in range(3):
                dic_temp = {}
                dic_temp['q1'] = data[i]["question"]
                dic_temp['q2'] = data[i]['negative_ctxs']['text'][j][1]
                dic_temp['label'] = 'irrelevantne'
                data_list.append(dic_temp)
                #print(dic_temp)
    return data_list

class QuaraDataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print('dataset:')
        # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
        return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], self.data[i]['label']

# class QuaraDataset(Dataset):
#     def __init__(self, dataset_type, data, tokenizer, args):
#         super().__init__()
#         self.args = args
#         self.data = data
#         self.dataset_type = dataset_type
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, i):
#         # print('dataset:')
#         # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
#         return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], self.data[i]['label']

class NQDataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print('dataset:')
        # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
        return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], self.data[i]['label']

class NQDataset_fintune(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print('dataset:')
        # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
        label = 0
        if(self.data[i]['label'] == 'relevant'):
            label = 1
        else:
            label = 0
        return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], label

class ADhocDataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print('dataset:')
        # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
        return self.data[i]['q'] + '[SEP]' + self.data[i]['d'], self.data[i]['label']

class MLIDataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print('dataset:')
        # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
        label = 0
        return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], self.data[i]['label']

class MLIDataset_fintune(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print('dataset:')
        # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
        label = 0
        if(self.data[i]['label'] == 'relevant'):
            label = 1
        else:
            label = 0
        return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], label

class DiaDataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print('dataset:')
        # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
        return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], self.data[i]['label']

class DiaDataset_fintune(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        label = 0
        if(self.data[i]['label'] == 'relevant'):
            label = 1
        else:
            label = 0
        return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], label


class adhocataset_fintune(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        label = 0
        if(self.data[i]['label'] == 'relevant'):
            label = 1
        else:
            label = 0
        return self.data[i]['q'] + '[SEP]' + self.data[i]['d'], label

class BM25Dataset(Dataset):
    def __init__(self, dataset_type, data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print('dataset:')
        # print((self.data[i]['q1'],self.data[i]['q2']), self.data[i]['label'])
        return self.data[i]['q1'] + '[SEP]' + self.data[i]['q2'], self.data[i]['label']