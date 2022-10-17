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

def load_csv(filname,s1,s2,q):
    data = []
    df = pd.read_csv(filname)
    for item in df.iterrows():
        dic_temp = {}
        dic_temp['q1'] = item[1]['question1']
        dic_temp['q2'] = item[1]['question2']
        dic_temp['s1'] = s1
        dic_temp['s2'] = s2
        dic_temp['q'] = q
        if(item[1]['is_duplicate'] == 1):
            dic_temp['label'] = 'relevant'
        else:
            dic_temp['label'] = 'irrelevant'
        #dic_temp['label'] = item[1]['is_duplicate']
        data.append(dic_temp)
    return data

def load_msrp(filname,s1,s2,q):
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
            dic_temp['q'] = q
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
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

def load_paws(filname,s1,s2,q):
    data = []
    df = pd.read_csv(filname,sep='\t')
    for item in df.iterrows():
        dic_temp = {}
        dic_temp['q1'] = item[1]['sentence1']
        dic_temp['q2'] = item[1]['sentence2']
        dic_temp['s1'] = s1
        dic_temp['s2'] = s2
        dic_temp['q'] = q
        if(item[1]['label'] == 1):
            dic_temp['label'] = 'relevant'
        else:
            dic_temp['label'] = 'irrelevant'
        #dic_temp['label'] = item[1]['is_duplicate']
        # print(dic_temp)
        data.append(dic_temp)
    return data

def load_parade(filename,s1,s2,q):
    data = []
    with open(filename,'r') as files:
        lines = files.readlines()
        for line in lines:
            line = line[:-1]
            line = line.split('\t')
            dic_temp = {}
            dic_temp['q1'] = line[3]
            dic_temp['q2'] = line[4]
            dic_temp['label'] = line[1]
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            if(line[1] == '1'):
                dic_temp['label'] = 'relevant'
            else:
                dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def is_irrelevantrmal_word(seq):
    for word in seq:
        if(len(word) > 50 or word == '余常聞之'):
            return False
    return True
def load_json_nq(filname,s1,s2,q):
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
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                dic_temp['label'] = 'relevant'
                data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            dic_temp['q2'] = dic[index+1]['positive_ctxs'][0]['text']
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            dic_temp['q2'] = dic[index]['negative_ctxs'][0]['text']
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
            if(len(dic[index]['negative_ctxs']) > 1):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['negative_ctxs'][0]['text']
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                dic_temp['label'] = 'irrelevant'
                data.append(dic_temp)
    return data

def load_json_nq_mrr(filname,s1,s2,q):
    data = []
    truth_dic = {}
    count = 0
    with open(filname, 'r') as load_f:
        dic = json.load(load_f)
        for index in range(len(dic)-1):
            if(count > 5000):
                break
            truch_list = []
            count_index = 0
            for pos_idx in range(len(dic[index]['positive_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['positive_ctxs'][pos_idx]['text']
                dic_temp['label'] = 'relevant'
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                truch_list.append(dic_temp['q2_num'])
                dic_temp['q'] = q
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                data.append(dic_temp)
                count_index += 1
            for neg_idx in range(len(dic[index]['negative_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['negative_ctxs'][neg_idx]['text']
                dic_temp['label'] = 'irrelevant'
                dic_temp['q'] = q
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                data.append(dic_temp)
                count_index += 1
            for hard_neg_idx in range(len(dic[index]['hard_negative_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['hard_negative_ctxs'][hard_neg_idx]['text']
                dic_temp['label'] = 'irrelevant'
                dic_temp['q'] = q
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                data.append(dic_temp)
                count_index += 1
            truth_dic[str(index)] = truch_list
            count += 1
    return data,truth_dic

def load_json_trivia(filname,s1,s2,q):
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
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            if(len(dic[index+1]['positive_ctxs']) == 0):
                continue
            dic_temp['q2'] = dic[index+1]['positive_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            if (len(dic[index]['hard_negative_ctxs']) == 0):
                continue
            dic_temp['q2'] = dic[index]['hard_negative_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            data.append(dic_temp)
            if(len(dic[index]['hard_negative_ctxs']) > 1):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['hard_negative_ctxs'][1]['text']
                dic_temp['label'] = 'irrelevant'
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                data.append(dic_temp)
    return data

def load_json_squad(filname,s1,s2,q):
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
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            if(len(dic[index+1]['negative_ctxs']) == 0):
                continue
            dic_temp['q2'] = dic[index]['negative_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            data.append(dic_temp)
            dic_temp = {}
            dic_temp['q1'] = dic[index]['question']
            if (len(dic[index]['hard_negative_ctxs']) == 0):
                continue
            dic_temp['q2'] = dic[index]['hard_negative_ctxs'][0]['text']
            dic_temp['label'] = 'irrelevant'
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            data.append(dic_temp)
            if(len(dic[index]['hard_negative_ctxs']) > 1):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                if (len(dic[index]['hard_negative_ctxs']) == 0):
                    continue
                dic_temp['q2'] = dic[index]['hard_negative_ctxs'][0]['text']
                dic_temp['label'] = 'irrelevant'
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                data.append(dic_temp)
    return data


def load_json_trivia_mrr(filname,s1,s2,q):
    data = []
    truth_dic = {}
    with open(filname, 'r') as load_f:
        dic = json.load(load_f)
        for index in range(len(dic)-1):
            truch_list = []
            count_index = 0
            for pos_idx in range(len(dic[index]['positive_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['positive_ctxs'][pos_idx]['text']
                dic_temp['label'] = 'relevant'
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                truch_list.append(dic_temp['q2_num'])
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                data.append(dic_temp)
                count_index += 1
            for neg_idx in range(len(dic[index]['negative_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index+1]['positive_ctxs'][neg_idx]['text']
                dic_temp['label'] = 'irrelevant'
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                data.append(dic_temp)
                count_index += 1
            for hard_neg_idx in range(len(dic[index]['hard_negative_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['hard_negative_ctxs'][hard_neg_idx]['text']
                dic_temp['label'] = 'irrelevant'
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                data.append(dic_temp)
                count_index += 1
            truth_dic[str(index)] = truch_list
    return data,truth_dic


def load_json_trivia_mrr_task(filname,s1,s2,q):
    data = []
    truth_dic = {}
    with open(filname, 'r') as load_f:
        dic = json.load(load_f)
        for index in range(len(dic)-1):
            try:
                truch_list = []
                #count_index = 0
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['positive_ctxs'][0]['text']
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['label'] = 'relevant'
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(0)
                truch_list.append(dic_temp['q2_num'])
                dic_temp['q'] = q
                data.append(dic_temp)
                #count_index += 1

                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic_temp['q1']
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['label'] = 'irrelevant'
                dic_temp['q'] = q
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(1)
                #truch_list.append(dic_temp['q2_num'])
                data.append(dic_temp)
                truth_dic[str(index)] = truch_list
            except:
                pass

    return data,truth_dic

def load_json_trivia_mrr_few(filname,s1,s2,q):
    data = []
    truth_dic = {}
    count = 0
    with open(filname, 'r') as load_f:
        dic = json.load(load_f)
        for index in range(len(dic)-1):
            if(count > 1000):
                break
            truch_list = []
            count_index = 0
            for pos_idx in range(len(dic[index]['positive_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['positive_ctxs'][pos_idx]['text']
                dic_temp['label'] = 'relevant'
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                truch_list.append(dic_temp['q2_num'])
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                data.append(dic_temp)
                count_index += 1
            for neg_idx in range(len(dic[index]['negative_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index+1]['positive_ctxs'][neg_idx]['text']
                dic_temp['label'] = 'irrelevant'
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                data.append(dic_temp)
                count_index += 1
            for hard_neg_idx in range(len(dic[index]['hard_negative_ctxs'])):
                dic_temp = {}
                dic_temp['q1'] = dic[index]['question']
                dic_temp['q2'] = dic[index]['hard_negative_ctxs'][hard_neg_idx]['text']
                dic_temp['label'] = 'irrelevant'
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                dic_temp['q1_num'] = str(index)
                dic_temp['q2_num'] = str(count_index)
                data.append(dic_temp)
                count_index += 1
            truth_dic[str(index)] = truch_list
            count += 1
    return data,truth_dic

def load_adhoc(filname,s1,s2,qe):
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
    train_pos_list = []
    train_neg_list = []
    dev_pos_list = []
    dev_neg_list = []
    test_pos_list = []
    test_neg_list = []
    dic_qels_dev = {}
    dic_qels_test = {}
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
            if ('train' in file and file != 'train'):
                path = root+'/'+file
                with open(path,'r') as train:
                    for line in train.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[1]]
                            label = dic_label[line[0]][line[1]]
                            label_flag = False
                            if(label == '0'):
                                label = 'irrelevant'
                                train_neg += 1
                            else:
                                label = 'relevant'
                                train_pos += 1
                                label_flag = True
                            dic_temp = {}
                            dic_temp['q1'] = q
                            dic_temp['q2'] = d
                            dic_temp['q'] = qe
                            dic_temp['s1'] = s1
                            dic_temp['s2'] = s2
                            dic_temp['q1_real'] = line[0]
                            dic_temp['q2_real'] = line[1]
                            dic_temp['label'] = label
                            if label_flag:
                                train_pos_list.append(dic_temp)
                            else:
                                train_neg_list.append(dic_temp)
                            #print(dic_temp)
                            train_data.append(dic_temp)
                        except:
                            pass
            if ('valid' in file and file != 'valid'):
                path = root+'/'+file
                with open(path,'r') as valid:
                    for line in valid.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[2]]
                            label = line[4]
                            label_flag = False
                            if(label == '0'):
                                label = 'irrelevant'
                                dev_neg += 1
                            else:
                                label = 'relevant'
                                dev_pos += 1
                                label_flag = True
                            # else:
                            #     continue
                            dic_temp = {}
                            dic_temp['q1'] = q
                            dic_temp['q2'] = d
                            dic_temp['q'] = qe
                            dic_temp['s1'] = s1
                            dic_temp['s2'] = s2
                            dic_temp['q1_real'] = line[0]
                            dic_temp['q2_real'] = line[2]
                            dic_temp['label'] = label
                            if(line[0] in dic_qels_dev.keys()):
                                dic_qels_dev[line[0]].append((line[2],int(line[4])))
                            else:
                                dic_qels_dev[line[0]] = []
                                dic_qels_dev[line[0]].append((line[2], int(line[4])))
                            if label_flag:
                                dev_pos_list.append(dic_temp)
                            else:
                                dev_neg_list.append(dic_temp)
                            dev_data.append(dic_temp)
                        except:
                            pass
            if ('test' in file and file != 'test'):
                path = root+'/'+file
                with open(path,'r') as test:
                    for line in test.readlines():
                        line = line[:-1].split()
                        try:
                            q = dic_query[line[0]]
                            d = dic_dataset[line[2]]
                            label = line[4]
                            label_flag = False
                            if(label == '0'):
                                label = 'irrelevant'
                                test_neg += 1
                            else:
                                label = 'relevant'
                                test_pos += 1
                                label_flag = True
                            # else:
                            #     continue
                            dic_temp = {}
                            dic_temp['q1'] = q
                            dic_temp['q2'] = d
                            dic_temp['q'] = qe
                            dic_temp['s1'] = s1
                            dic_temp['s2'] = s2
                            dic_temp['q1_real'] = line[0]
                            dic_temp['q2_real'] = line[2]
                            dic_temp['label'] = label
                            test_data.append(dic_temp)
                            if(line[0] in dic_qels_test.keys()):
                                dic_qels_test[line[0]].append((line[2],int(line[4])))
                            else:
                                dic_qels_test[line[0]] = []
                                dic_qels_test[line[0]].append((line[2], int(line[4])))
                            if(label_flag):
                                test_pos_list.append(dic_temp)
                            else:
                                test_neg_list.append(dic_temp)
                        except:
                            pass
    print(train_pos)
    print(train_neg)
    print(dev_pos)
    print(dev_neg)
    print(test_pos)
    print(test_neg)
    # random.shuffle(train_neg_list)
    # train_neg_list = train_neg_list[:train_pos]
    # random.shuffle(dev_pos_list)
    # dev_neg_list = dev_neg_list[:dev_pos]
    # random.shuffle(test_pos_list)
    # test_neg_list = test_neg_list[:test_pos]

    train_data = train_pos_list + train_neg_list
    dev_data = dev_pos_list + dev_neg_list
    test_data = test_pos_list + test_neg_list

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    for key in dic_qels_dev.keys():
        dic_qels_dev[key] = sorted(dic_qels_dev[key], key=lambda item: item[1],reverse = True)
    return train_data,dev_data,test_data,dic_qels_dev,dic_qels_test

def load_adhoc_mq(filname,s1,s2,qe):
    train_data = []
    dev_data = []
    test_data = []
    dic_dataset = {}
    train_pos_list = []
    train_neg_list = []
    dev_pos_list = []
    dev_neg_list = []
    test_pos_list = []
    test_neg_list = []
    train_pos = 0
    train_neg = 0
    dev_pos = 0
    dev_neg = 0
    test_pos = 0
    test_neg = 0
    dic_qels_dev = {}
    dic_qels_test = {}
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
            try:
                if ('train' in file and file != 'train' and '1' in file):
                    path = root+'/'+file
                    with open(path,'r') as train:
                        for line in train.readlines():
                            line = line[:-1].split()
                            try:
                                q = dic_query[line[0]]
                                d = dic_dataset[line[1]]
                                label = dic_label[line[0]][line[1]]
                                label_flag = False
                                if(label == '0'):
                                    label = 'irrelevant'
                                    train_neg += 1
                                else:
                                    label = 'relevant'
                                    train_pos += 1
                                    label_flag = True
                                dic_temp = {}
                                dic_temp['q1'] = q
                                dic_temp['q2'] = d
                                dic_temp['q'] = qe
                                dic_temp['s1'] = s1
                                dic_temp['s2'] = s2
                                dic_temp['q1_real'] = line[0]
                                dic_temp['q2_real'] = line[1]
                                dic_temp['label'] = label
                                #print(dic_temp)
                                train_data.append(dic_temp)
                                if label_flag:
                                    train_pos_list.append(dic_temp)
                                else:
                                    train_neg_list.append(dic_temp)
                                # print(dic_temp)
                            except:
                                pass
                if ('valid' in file and file != 'valid' and '1' in file):
                    path = root+'/'+file
                    with open(path,'r') as valid:
                        for line in valid.readlines():
                            line = line[:-1].split()
                            try:
                                q = dic_query[line[0]]
                                d = dic_dataset[line[2]]
                                label = line[4]
                                label_flag = False
                                if(label == '0'):
                                    label = 'irrelevant'
                                    dev_neg += 1
                                else:
                                    label = 'relevant'
                                    dev_pos += 1
                                    label_flag = True
                                # else:
                                #     continue
                                dic_temp = {}
                                dic_temp['q1'] = q
                                dic_temp['q2'] = d
                                dic_temp['q'] = qe
                                dic_temp['s1'] = s1
                                dic_temp['s2'] = s2
                                dic_temp['q1_real'] = line[0]
                                dic_temp['q2_real'] = line[2]
                                dic_temp['label'] = label
                                dev_data.append(dic_temp)
                                if(line[0] in dic_qels_dev.keys()):
                                    dic_qels_dev[line[0]].append((line[2],int(line[4])))
                                else:
                                    dic_qels_dev[line[0]] = []
                                    dic_qels_dev[line[0]].append((line[2], int(line[4])))
                                if label_flag:
                                    dev_pos_list.append(dic_temp)
                                else:
                                    dev_neg_list.append(dic_temp)
                            except:
                                pass
                if ('test' in file and file != 'test' and '1' in file):
                    path = root+'/'+file
                    with open(path,'r') as test:
                        for line in test.readlines():
                            line = line[:-1].split()
                            try:
                                q = dic_query[line[0]]
                                d = dic_dataset[line[2]]
                                label = line[4]
                                label_flag = False
                                if(label == '0'):
                                    label = 'irrelevant'
                                    test_neg += 1
                                else:
                                    label = 'relevant'
                                    test_pos += 1
                                    label_flag = True
                                # else:
                                #     continue
                                dic_temp = {}
                                dic_temp['q1'] = q
                                dic_temp['q2'] = d
                                dic_temp['q'] = qe
                                dic_temp['s1'] = s1
                                dic_temp['s2'] = s2
                                dic_temp['q1_real'] = line[0]
                                dic_temp['q2_real'] = line[2]
                                dic_temp['label'] = label
                                test_data.append(dic_temp)
                                if(line[0] in dic_qels_test.keys()):
                                    dic_qels_test[line[0]].append((line[2],int(line[4])))
                                else:
                                    dic_qels_test[line[0]] = []
                                    dic_qels_test[line[0]].append((line[2], int(line[4])))
                                if label_flag:
                                    test_pos_list.append(dic_temp)
                                else:
                                    test_neg_list.append(dic_temp)
                            except:
                                pass
            except:
                pass
    print(train_pos)
    print(train_neg)
    print(dev_pos)
    print(dev_neg)
    print(test_pos)
    print(test_neg)
    # random.shuffle(train_neg_list)
    # train_neg_list = train_neg_list[:train_pos]
    # random.shuffle(dev_pos_list)
    # dev_neg_list = dev_neg_list[:dev_pos]
    # random.shuffle(test_pos_list)
    # test_neg_list = test_neg_list[:test_pos]

    train_data = train_pos_list + train_neg_list
    dev_data = dev_pos_list + dev_neg_list
    test_data = test_pos_list + test_neg_list
    # random.shuffle(train_data)
    # random.shuffle(dev_data)
    # random.shuffle(test_data)
    for key in dic_qels_dev.keys():
        dic_qels_dev[key] = sorted(dic_qels_dev[key], key=lambda item: item[1],reverse = True)
    return train_data,dev_data,test_data,dic_qels_dev,dic_qels_test

def load_adhoc_mq_train(filname, s1, s2, qe):
    train_data = []
    dev_data = []
    test_data = []
    dic_dataset = {}
    train_pos_list = []
    train_neg_list = []
    dev_pos_list = []
    dev_neg_list = []
    test_pos_list = []
    test_neg_list = []
    train_pos = 0
    train_neg = 0
    dev_pos = 0
    dev_neg = 0
    test_pos = 0
    test_neg = 0
    dic_qels_dev = {}
    dic_qels_test = {}
    with open(filname + 'dataset.txt', 'r') as dataset:
        for line in dataset.readlines():
            line = line.split('\t')
            seq = line[3][:-1]
            idx = seq.find('/')
            if (not idx == -1):
                seq = seq[idx + 2:]
            dic_dataset[line[1]] = seq
    dic_query = {}
    with open(filname + 'queries.txt', 'r') as queries:
        for line in queries.readlines():
            line = line.split('\t')
            seq = line[2][:-1]
            dic_query[line[1]] = seq
    dic_label = {}
    with open(filname + 'qrels', 'r') as qrels:
        for line in qrels.readlines():
            line = line.split('\t')
            # print(line)
            q = line[0]
            d = line[2]
            label = line[3][:-1]
            if (q in dic_label.keys()):
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
            # print(file)
            try:
                if ('train' in file):
                    path = root + '/' + file
                    with open(path, 'r') as train:
                        for line in train.readlines():
                            line = line[:-1].split()
                            try:
                                q = dic_query[line[0]]
                                d = dic_dataset[line[1]]
                                label = dic_label[line[0]][line[1]]
                                label_flag = False
                                if (label == '0'):
                                    label = 'irrelevant'
                                    train_neg += 1
                                else:
                                    label = 'relevant'
                                    train_pos += 1
                                    label_flag = True
                                dic_temp = {}
                                dic_temp['q1'] = q
                                dic_temp['q2'] = d
                                dic_temp['q'] = qe
                                dic_temp['s1'] = s1
                                dic_temp['s2'] = s2
                                dic_temp['q1_real'] = line[0]
                                dic_temp['q2_real'] = line[1]
                                dic_temp['label'] = label
                                # print(dic_temp)
                                train_data.append(dic_temp)
                                if label_flag:
                                    train_pos_list.append(dic_temp)
                                else:
                                    train_neg_list.append(dic_temp)
                                # print(dic_temp)
                            except:
                                pass
                if ('valid' in file):
                    path = root + '/' + file
                    with open(path, 'r') as valid:
                        for line in valid.readlines():
                            line = line[:-1].split()
                            try:
                                q = dic_query[line[0]]
                                d = dic_dataset[line[2]]
                                label = line[4]
                                label_flag = False
                                if (label == '0'):
                                    label = 'irrelevant'
                                    dev_neg += 1
                                else:
                                    label = 'relevant'
                                    dev_pos += 1
                                    label_flag = True
                                # else:
                                #     continue
                                dic_temp = {}
                                dic_temp['q1'] = q
                                dic_temp['q2'] = d
                                dic_temp['q'] = qe
                                dic_temp['s1'] = s1
                                dic_temp['s2'] = s2
                                dic_temp['q1_real'] = line[0]
                                dic_temp['q2_real'] = line[2]
                                dic_temp['label'] = label
                                dev_data.append(dic_temp)
                                if (line[0] in dic_qels_dev.keys()):
                                    dic_qels_dev[line[0]].append((line[2], int(line[4])))
                                else:
                                    dic_qels_dev[line[0]] = []
                                    dic_qels_dev[line[0]].append((line[2], int(line[4])))
                                if label_flag:
                                    dev_pos_list.append(dic_temp)
                                else:
                                    dev_neg_list.append(dic_temp)
                            except:
                                pass
                if ('test' in file):
                    path = root + '/' + file
                    with open(path, 'r') as test:
                        for line in test.readlines():
                            line = line[:-1].split()
                            try:
                                q = dic_query[line[0]]
                                d = dic_dataset[line[2]]
                                label = line[4]
                                label_flag = False
                                if (label == '0'):
                                    label = 'irrelevant'
                                    test_neg += 1
                                else:
                                    label = 'relevant'
                                    test_pos += 1
                                    label_flag = True
                                # else:
                                #     continue
                                dic_temp = {}
                                dic_temp['q1'] = q
                                dic_temp['q2'] = d
                                dic_temp['q'] = qe
                                dic_temp['s1'] = s1
                                dic_temp['s2'] = s2
                                dic_temp['q1_real'] = line[0]
                                dic_temp['q2_real'] = line[2]
                                dic_temp['label'] = label
                                test_data.append(dic_temp)
                                if (line[0] in dic_qels_test.keys()):
                                    dic_qels_test[line[0]].append((line[2], int(line[4])))
                                else:
                                    dic_qels_test[line[0]] = []
                                    dic_qels_test[line[0]].append((line[2], int(line[4])))
                                if label_flag:
                                    test_pos_list.append(dic_temp)
                                else:
                                    test_neg_list.append(dic_temp)
                            except:
                                pass
            except:
                pass
    print(train_pos)
    print(train_neg)
    print(dev_pos)
    print(dev_neg)
    print(test_pos)
    print(test_neg)
    random.shuffle(train_neg_list)
    train_neg_list = train_neg_list[:train_pos]
    random.shuffle(dev_pos_list)
    dev_neg_list = dev_neg_list[:dev_pos]
    random.shuffle(test_pos_list)
    test_neg_list = test_neg_list[:test_pos]

    train_data = train_pos_list + train_neg_list
    dev_data = dev_pos_list + dev_neg_list
    test_data = test_pos_list + test_neg_list
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    return train_data[:20000],dev_data[:5000],test_data[:5000]

def load_mli(filname,s1,s2,q):
    data = []
    with open(filname,'r') as mli:
        json_data = json.load(mli)
        for item in json_data:
            dic_temp = {}
            dic_temp['q1'] = item['sentence1']
            dic_temp['q2'] = item['sentence2']
            dic_temp['label'] = item['annotator_labels']
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
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


def load_url(filename,s1,s2,q):
    data = []
    count = 0
    with open(filename,'r') as files:
        lines = files.readlines()
        for line in lines:
            if(count == 0):
                count += 1
                continue
            line = line[:-1]
            line = line.split('\t')
            dic_temp = {}
            dic_temp['q1'] = line[1]
            dic_temp['q2'] = line[2]
            dic_temp['label'] = line[3]
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            if(line[3] == '0'):
                dic_temp['label'] = 'irrelevant'
            else:
                dic_temp['label'] = 'relevant'
            data.append(dic_temp)
    return data

def load_ali(filname,s1,s2,q):
    data = []
    with open(filname,'r') as mli:
        for item in json_lines.reader(mli):
            dic_temp = {}
            dic_temp['q1'] = item['context']
            dic_temp['q2'] = item['hypothesis']
            dic_temp['label'] = item['label']
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            if(item['label'] == 'e'):
                dic_temp['label'] = 'relevant'
            else:
                dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_snli(filname,s1,s2,q):
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
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
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

def load_scitail(filename,s1,s2,q):
    data = []
    with open(filename,'r') as scitail:
        lines = scitail.readlines()
        for line in lines:
            line = line[:-1].split('\t')
            dic_temp = {}
            dic_temp['q1'] = line[0]
            dic_temp['q2'] = line[1]
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            if(line[2] == 'entails'):
                dic_temp['label'] = 'relevant'
            else:
                dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_rte(filname,s1,s2,q):
    data = []
    with open(filname,'r') as mli:
        for item in json_lines.reader(mli):
            dic_temp = {}
            dic_temp['q1'] = item['premise']
            dic_temp['q2'] = item['hypothesis']
            dic_temp['label'] = item['label']
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            if(item['label'] == 'entailment'):
                dic_temp['label'] = 'relevant'
            else:
                dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
    return data

def load_qnli(filname,s1,s2,q):
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
            dic_tem['s1'] = s1
            dic_tem['s2'] = s2
            dic_tem['q'] = q
            if(line[3] == 'entailment'):
                dic_tem['label'] = 'relevant'
            else:
                dic_tem['label'] = 'irrelevant'
            data.append(dic_tem)
    return data

def load_sick(filname,s1,s2,q):
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
            dic_tem['s1'] = s1
            dic_tem['s2'] = s2
            dic_tem['q'] = q
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

def load_dialogue(filname,s1,s2,q):
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
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q'] = q
            last_q2 = line[1]
            dic_temp['label'] = 'relevant'
            pos += 1
            data.append(dic_temp)
            for i in range(1): #随机采样负样本
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
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q'] = q
                data.append(dic_temp)
                neg += 1
    print(pos)
    print(neg)
    return data

def load_dailydialogue(filname,s1,s2,q):
    data = []
    with open(filname,'r') as reddit:
        lines = reddit.readlines()
        last_q2 = ''
        first_line = False
        len_all = len(lines)-1
        for idx in range(len(lines)-1): ##确定这一行是对话轮数最多的那行
            if(not '__eou__' in lines[idx+1]):
                line = lines[idx][:-1].split('__eou__')
                q1 = line[0]
                round_count = 0
                for idx_line in range(1,len(line)):
                    dic_temp = {}
                    dic_temp['q1'] = q1
                    dic_temp['q2'] = line[idx_line]
                    dic_temp['q'] = q
                    dic_temp['s1'] = s1
                    dic_temp['s2'] = s2
                    dic_temp['label'] = 'relevant'
                    data.append(dic_temp)
                    #print(dic_temp)
                    # 采样负样本
                    dic_temp = {}
                    dic_temp['q1'] = q1
                    #neg_index = random.randint(0, len(lines) - 1)
                    neg_index = 0
                    if(idx < len(lines)-1):
                        neg_index = idx+1
                    #for neg_index in range(len(lines)-1):
                    while 1:
                        if (not '__eou__' in lines[neg_index]):
                            neg_index = (neg_index+1) % len_all
                            continue
                        neg_line = (lines[neg_index].split('__eou__'))[0]
                        if neg_line == q1:
                            neg_index = (neg_index + 1) % len_all
                            continue
                        neg_q2 = (lines[neg_index].split('__eou__'))[1]
                        dic_temp['q2'] = neg_q2
                        dic_temp['q'] = q
                        dic_temp['s1'] = s1
                        dic_temp['s2'] = s2
                        dic_temp['label'] = 'irrelevant'
                        data.append(dic_temp)
                        break
                        #print(dic_temp)
                    q1 = line[idx_line]
                    round_count += 1
                    if(round_count == 3):
                        break
    return data


# def build_answer_base(filname):
#     answer_base = {}
#     truch_dic = {}
#     with open(filname,'r') as reddit:
#         lines = reddit.readlines()
#         last_q2 = ''
#         first_line = False
#         len_all = len(lines) - 1
#         for idx in range(len(lines)-1): ##确定这一行是对话轮数最多的那行
#             if(not '__eou__' in lines[idx+1]):
#                 line = lines[idx][:-1].split('__eou__')
#                 q1 = line[0]
#                 round_count = 0
#                 for idx_line in range(1,len(line)):
#                     #answer_base.add(line[idx_line])
#                     if(q1 in answer_base.keys()):
#                         answer_base[q1].append(line[idx_line])
#                     else:
#                         answer_base[q1] = [line[idx_line]]
#                     #print(answer_base[q1])
#                     #neg_index = random.randint(0, len(lines) - 1)
#                     neg_index = 0
#                     if(idx < len(lines)-1):
#                         neg_index = idx+1
#                     #for neg_index in range(len(lines)-1):
#                     neg_num = 0
#                     while 1:
#                         if (not '__eou__' in lines[neg_index]): ## 循环寻找满足条件的负样本，每个query找49个
#                             neg_index = (neg_index+1) % len_all
#                             continue
#                         neg_line = (lines[neg_index].split('__eou__'))[0]
#                         if neg_line == q1:
#                             neg_index = (neg_index + 1) % len_all
#                             continue
#                         neg_q2 = (lines[neg_index].split('__eou__'))[1]
#                         if(neg_q2 in answer_base[q1]):
#                             neg_index = (neg_index + 1) % len_all
#                             continue
#                         #answer_base.add(neg_q2)
#                         answer_base[q1].append(neg_q2)
#                         neg_index = (neg_index + 1) % len_all
#                         neg_num += 1
#                         if(neg_num >= 9):
#                             break
#                         #print(dic_temp)
#                     q1 = line[idx_line]
#                     round_count += 1
#                     if(round_count == 3):
#                         break
#     return answer_base
def build_answer_base(filname):
    answer_base = {}
    truch_dic = {}
    with open(filname,'r') as reddit:
        lines = reddit.readlines()
        last_q2 = ''
        first_line = False
        len_all = len(lines) - 1
        for idx in range(len(lines)-1): ##确定这一行是对话轮数最多的那行
            if(not '__eou__' in lines[idx+1]):
                #line = lines[idx][:-1].split('__eou__')
                line = re.split('__eou__|\t', lines[idx][:-1])
                q1 = line[0]
                round_count = 0
                for idx_line in range(1,len(line)):
                    #answer_base.add(line[idx_line])
                    if(q1 in answer_base.keys()):
                        answer_base[q1].append(line[idx_line])
                    else:
                        answer_base[q1] = [line[idx_line]]
                    #print(answer_base[q1])
                    #neg_index = random.randint(0, len(lines) - 1)
                    neg_index = 0
                    if(idx < len(lines)-1):
                        neg_index = idx+1
                    #for neg_index in range(len(lines)-1):
                    neg_num = 0
                    while 1:
                        if (not '__eou__' in lines[neg_index]): ## 循环寻找满足条件的负样本，每个query找49个
                            neg_index = (neg_index+1) % len_all
                            continue
                        #neg_line = (lines[neg_index].split('__eou__'))[0]
                        neg_line = re.split('__eou__|\t', lines[neg_index][:-1])[0]
                        if neg_line == q1:
                            neg_index = (neg_index + 1) % len_all
                            continue
                        #neg_q2 = (lines[neg_index].split('__eou__'))[1]
                        neg_q2 = re.split('__eou__|\t', lines[neg_index][:-1])[1]
                        if(neg_q2 in answer_base[q1]):
                            neg_index = (neg_index + 1) % len_all
                            continue
                        #answer_base.add(neg_q2)
                        answer_base[q1].append(neg_q2)
                        neg_index = (neg_index + 1) % len_all
                        neg_num += 1
                        if(neg_num >= 9):
                            break
                        #print(dic_temp)
                    q1 = line[idx_line]
                    round_count += 1
                    if(round_count == 3):
                        break
    return answer_base

import re
def load_dailydialogue_mrr(filname,answer_base,s1,s2,q):
    data = []
    truch_dic = {}
    q1_num = 0
    with open(filname,'r') as reddit:
        lines = reddit.readlines()
        last_q2 = ''
        first_line = False
        len_all = len(lines) - 1
        for idx in range(len(lines)-1): ##确定这一行是对话轮数最多的那行
            if(not '__eou__' in lines[idx+1]):
                #line = lines[idx][:-1].split('__eou__')
                line = re.split('__eou__|\t', lines[idx][:-1])
                q1 = line[0]
                round_count = 0
                for idx_line in range(1,len(line)):
                    dic_temp = {}
                    dic_temp['q'] = q
                    dic_temp['s1'] = s1
                    dic_temp['s2'] = s2
                    dic_temp['q1'] = q1
                    dic_temp['q1_num'] = str(q1_num)
                    dic_temp['q2'] = line[idx_line]
                    #print(answer_base[q1][])
                    dic_temp['q2_num'] = str(answer_base[q1].index(dic_temp['q2']))
                    dic_temp['label'] = 'relevant'
                    truch_dic[dic_temp['q1_num']] = dic_temp['q2_num']
                    data.append(dic_temp)
                    ## 构造负样本
                    for answer in answer_base[q1]:
                        dic_temp = {}
                        dic_temp['q'] = q
                        dic_temp['q1'] = q1
                        dic_temp['s1'] = s1
                        dic_temp['s2'] = s2
                        dic_temp['q1_num'] = str(q1_num)
                        dic_temp['q2'] = answer
                        dic_temp['q2_num'] = str(answer_base[q1].index(dic_temp['q2']))
                        if(dic_temp['q2_num'] == truch_dic[dic_temp['q1_num']]):
                            continue
                        dic_temp['label'] = 'irrelevant'
                        data.append(dic_temp)
                    q1 = line[idx_line]
                    q1_num += 1
                    round_count += 1
                    if(round_count == 3):
                        break
    return data,truch_dic

# def load_dailydialogue_mrr(filname,answer_base,s1,s2,q):
#     data = []
#     truch_dic = {}
#     q1_num = 0
#     with open(filname,'r') as reddit:
#         lines = reddit.readlines()
#         last_q2 = ''
#         first_line = False
#         len_all = len(lines) - 1
#         for idx in range(len(lines)-1): ##确定这一行是对话轮数最多的那行
#             if(not '__eou__' in lines[idx+1]):
#                 line = lines[idx][:-1].split('__eou__')
#                 q1 = line[0]
#                 round_count = 0
#                 for idx_line in range(1,len(line)):
#                     dic_temp = {}
#                     dic_temp['q'] = q
#                     dic_temp['s1'] = s1
#                     dic_temp['s2'] = s2
#                     dic_temp['q1'] = q1
#                     dic_temp['q1_num'] = str(q1_num)
#                     dic_temp['q2'] = line[idx_line]
#                     #print(answer_base[q1][])
#                     dic_temp['q2_num'] = str(answer_base[q1].index(dic_temp['q2']))
#                     dic_temp['label'] = 'relevant'
#                     truch_dic[dic_temp['q1_num']] = dic_temp['q2_num']
#                     data.append(dic_temp)
#                     ## 构造负样本
#                     for answer in answer_base[q1]:
#                         dic_temp = {}
#                         dic_temp['q'] = q
#                         dic_temp['s1'] = s1
#                         dic_temp['s2'] = s2
#                         dic_temp['q1'] = q1
#                         dic_temp['q1_num'] = str(q1_num)
#                         dic_temp['q2'] = answer
#                         dic_temp['q2_num'] = str(answer_base[q1].index(dic_temp['q2']))
#                         if(dic_temp['q2_num'] == truch_dic[dic_temp['q1_num']]):
#                             continue
#                         dic_temp['label'] = 'irrelevant'
#                         data.append(dic_temp)
#                     q1 = line[idx_line]
#                     q1_num += 1
#                     round_count += 1
#                     if(round_count == 3):
#                         break
#     return data,truch_dic

def load_dialogue_mrr(filname,s1,s2,q):
    data = []
    pos = 0
    neg = 0
    truth_dic ={}
    q_jilu = []
    with open(filname,'r') as reddit:
        lines = reddit.readlines()
        last_q2 = ''
        len_all = len(lines)
        for line in lines:
            pos_index = lines.index(line)
            if('deleted' in line):
                continue
            line = line[:-1].split('__eou__')
            if(line[1] == last_q2):
                continue
            if(line[0] in q_jilu):
                continue
            q2_num = 0
            dic_temp = {}
            dic_temp['q1'] = line[0]
            dic_temp['q2'] = line[1]
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q1_num'] = str(pos_index)
            dic_temp['q2_num'] = str(q2_num)
            dic_temp['q'] = q
            last_q2 = line[1]
            dic_temp['label'] = 'relevant'
            truth_dic[dic_temp['q1_num']] = dic_temp['q2_num']
            data.append(dic_temp)
            q2_num += 1
            neg_index = (pos_index + 1) % len_all
            while q2_num < 10:
                neg_line = (lines[neg_index].split('__eou__'))[0]
                while neg_line == line[0]:
                    neg_index = (neg_index + 1) % len_all
                    neg_line = (lines[neg_index].split('__eou__'))[0]
                dic_temp = {}
                dic_temp['q1'] = line[0]
                dic_temp['q2'] = (lines[neg_index].split('__eou__'))[1]
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
                dic_temp['q1_num'] = str(pos_index)
                dic_temp['q2_num'] = str(q2_num)
                dic_temp['q'] = q
                dic_temp['label'] = 'irrelevant'
                data.append(dic_temp)
                q2_num += 1
                neg_index = (neg_index + 1) % len_all
    return data,truth_dic
import joblib
def load_amazon(filname,s1,s2,q):
    data = []
    truth_dic = {}
    # with open(filname,'r') as amazon:
    #     dic = jsonlines.Reader(amazon)
    dic = joblib.load(filname)
    count = 0
    for item in dic:
        if(count > 5000):
            break
        # print(item['questionText'])
        # print(item['answers'][0]['answerText'])
        truth_dic[str(dic.index(item))] = []
        doc_num = 0
        for truth in item['answers']:
            dic_temp = {}
            dic_temp['q'] = q
            dic_temp['q1'] = item['questionText']
            dic_temp['q2'] = truth['answerText']
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q1_num'] = str(dic.index(item))
            dic_temp['q2_num'] = str(doc_num)
            dic_temp['label'] = 'relevant'
            truth_dic[dic_temp['q1_num'] ].append(dic_temp['q2_num'])
            data.append(dic_temp)
            doc_num += 1
        neg_num = 0
        pos_index = dic.index(item)
        neg_index = (pos_index +1) % (len(dic)-1)
        while neg_num < 30:
            item_neg = dic[neg_index]
            dic_temp = {}
            dic_temp['q'] = q
            dic_temp['s1'] = s1
            dic_temp['s2'] = s2
            dic_temp['q1'] = item['questionText']
            dic_temp['q2'] = item_neg['answers'][0]['answerText']
            dic_temp['q1_num'] = str(dic.index(item))
            dic_temp['q2_num'] = str(doc_num)
            dic_temp['label'] = 'irrelevant'
            data.append(dic_temp)
            doc_num += 1
            neg_num += 1
            neg_index = (neg_index + 1) % (len(dic) - 1)
        count += 1
    return data, truth_dic

def load_dia_movie(conversation,lines,s1,s2,q):
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
                dic_temp['q'] = q
                dic_temp['s1'] = s1
                dic_temp['s2'] = s2
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
                        dic_temp['q'] = q
                        dic_temp['s1'] = s1
                        dic_temp['s2'] = s2
                        data.append(dic_temp)
                        #print(dic_temp)
                    except:
                        pass
            except:
                pass
                # print(dic_temp)
    random.shuffle(data)
    return data[:50],data[50:-100000],data[-1:]

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

class MultiDataset(Dataset):
    def __init__(self, dataset_type, data):
        super().__init__()
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]['s1'] + '[SEP]' + self.data[i]['q1'] + '[SEP]' + self.data[i]['s2'] + '[SEP]' +self.data[i]['q2'], self.data[i]['label']


class MultiDataset_yes_no(Dataset):
    def __init__(self, dataset_type, data):
        super().__init__()
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if (self.data[i]['label'] == 'relevant'):
            label = 'yes'
        elif (self.data[i]['label'] == 'irrelevant'):
            label = 'no'

        return self.data[i]['s1'] + '[SEP]' + self.data[i]['q1'] + '[SEP]' + self.data[i]['s2'] + '[SEP]' +self.data[i]['q2'] + '[SEP]' + self.data[i]['q'], label

class MultiDataset_yes_no_ndcg(Dataset):
    def __init__(self, dataset_type, data):
        super().__init__()
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if (self.data[i]['label'] == 'relevant'):
            label = 'yes'
        elif (self.data[i]['label'] == 'irrelevant'):
            label = 'no'

        return self.data[i]['q1_real'] + '[NDCG]' + self.data[i]['q2_real'] + '[NDCG]' +self.data[i]['s1'] + '[SEP]' + self.data[i]['q1'] + '[SEP]' + self.data[i]['s2'] + '[SEP]' +self.data[i]['q2'] + '[SEP]' + self.data[i]['q'], label


class MultiDataset_yes_no_mrr(Dataset):
    def __init__(self, dataset_type, data):
        super().__init__()
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if (self.data[i]['label'] == 'relevant'):
            label = 'yes'
        elif (self.data[i]['label'] == 'irrelevant'):
            label = 'no'

        return self.data[i]['q1_num'] + '[MRR]' + self.data[i]['q2_num'] + '[MRR]' +self.data[i]['s1'] + '[SEP]' + self.data[i]['q1'] + '[SEP]' + self.data[i]['s2'] + '[SEP]' +self.data[i]['q2'] + '[SEP]' + self.data[i]['q'], label

class MultiDataset_yes_nov2(Dataset):
    def __init__(self, dataset_type, data):
        super().__init__()
        self.data = data
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if (self.data[i]['label'] == 'relevant'):
            label = 'yes'
        elif (self.data[i]['label'] == 'irrelevant'):
            label = 'no'
        return self.data[i]['s1']+ self.data[i]['q1'] + '[SEP]' + self.data[i]['s2'] + self.data[i]['q2'] + '[SEP]' + self.data[i]['q'], label


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