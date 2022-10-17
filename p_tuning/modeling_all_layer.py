import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join

import re

from transformers import AutoTokenizer

from p_tuning.models import get_embedding_layer, create_model
from data_utils.vocab import get_vocab_by_strategy, token_wrapper
from data_utils.dataset import load_file

from p_tuning.prompt_encoder import PromptEncoder_all_layer
from p_tuning.prompt_encoder_init import PromptEncoder
from mytransformers.src.transformers.models.bert import BertForMaskedLM

class PTune_bert_all_layer(torch.nn.Module):

    def __init__(self, args, device, template,q_prompt):
        super().__init__()
        self.args = args
        self.device = device

        # load relation templates
        self.relation_templates = dict(
            (d['relation'], d['template']) for d in load_file(join(self.args.data_dir, 'relations.jsonl')))

        # load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model_name else self.args.model_name
        #self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained('/root/xsc/prompt/LAMA/bert', use_fast=False)
        # load pre-trained model
        if 'megatron' in self.args.model_name and self.args.use_lm_finetune:
            raise RuntimeError("Can not apply args.use_lm_finetune=True on MegatronLM 11B.")
        self.model = BertForMaskedLM.from_pretrained('/root/xsc/prompt/LAMA/'+args.model_name)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune

        for name,param in self.model.named_parameters():
            if 'word_embeddings' in name:
                param.requires_grad = False

        self.embeddings = self.model.get_input_embeddings()

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        #self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))
        self.allowed_vocab_ids = set()
        if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
            template = (template[0], template[1], 0)
        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        #print('pseudo_token_id')
        #print(self.pseudo_token_id)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder_all_layer(self.template, self.hidden_size, self.tokenizer, self.device, args)
        self.prompt_encoder = self.prompt_encoder.to(device)
        self.prompt = 'good'
        self.q_prompt = q_prompt
        self.replace_embeds = 0
        self.blocked_indices = 0

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        #print("embed_input:")
        #print(queries_for_embedding)
        raw_embeds = self.embeddings(queries_for_embedding)
        #print(raw_embeds)

        # For using handcraft prompts
        if self.args.use_original_template:
            return raw_embeds

        self.blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        #print(blocked_indices)
        self.replace_embeds = self.prompt_encoder()
        # replace_embeds = torch.cat([self.prompt_encoder_q1(),self.prompt_encoder_q2,self.prompt_encoder_prompt],dim=0)
        # print(replace_embeds)
        # 把输入中prompt token位置的向量替换成lstm+mlp的输出向量
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, self.blocked_indices[bidx, i], :] = self.replace_embeds[i, :]
        #print(raw_embeds)
        return raw_embeds

    def get_query(self, x_h, prompt_tokens, x_t=None):
        # For using handcraft prompts
        if self.args.use_original_template:
            if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
                query = re.sub(r'\[Y\].*', '', self.relation_templates[self.args.relation_id].replace('[X]', x_h))
                return self.tokenizer(' ' + query)['input_ids']
            else:
                query = self.relation_templates[self.args.relation_id].replace('[X]', x_h).replace('[Y]',
                                                                                                   self.tokenizer.mask_token)
                return self.tokenizer(' ' + query)['input_ids']
        # For P-tuning
        if 'gpt' not in self.args.model_name and 'megatron' not in self.args.model_name:
            # BERT-style model
            x_h = x_h.split('[SEP]')
            # print(x_h[0])
            # print(x_h[1])
            tok_0 = self.tokenizer.tokenize(' ' + x_h[0])
            tok_1 = self.tokenizer.tokenize(' ' + x_h[1])
            tok_q = self.tokenizer.tokenize(self.q_prompt)
            if(len(tok_0) + len(tok_1) + len(tok_q) > 480):
                tok_1 = tok_1[:480-len(tok_0)-len(tok_q)]
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + self.tokenizer.convert_tokens_to_ids(tok_0)
                    + [self.tokenizer.sep_token_id]
                    + prompt_tokens * self.template[1]
                    + self.tokenizer.convert_tokens_to_ids(tok_1)
                    + [self.tokenizer.sep_token_id]
                    + self.tokenizer.convert_tokens_to_ids(tok_q)
                    + [self.tokenizer.mask_token_id]  # head entity
                    + [self.tokenizer.sep_token_id]
                    ]
        else:
            raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name))

    def get_query_s(self, x_h, prompt,x_all,first):
        if 'gpt' not in self.args.model_name and 'megatron' not in self.args.model_name:

            tok_0 = self.tokenizer.tokenize(prompt + x_h)
            tok_all = self.tokenizer.tokenize(x_all)
            if(len(tok_all) > 480):
                if(first == 1):
                    tok_q = x_all.split('[SEP]')[0]
                    tok_0 = tok_0[:480-len(self.tokenizer.tokenize(tok_q))]
            return [
                     self.tokenizer.convert_tokens_to_ids(tok_0)
                    ]
        else:
            raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name))


    def forward(self, x_hs, x_ts, return_candidates=False):
        bz = len(x_hs)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]

        queries = [torch.LongTensor(self.get_query(x_hs[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        label_ids = torch.LongTensor([self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_ts[i])) for i in range(bz)]).reshape((bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id
        inputs_embeds = self.embed_input(queries)

        label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
            1).to(self.device)  # bz * 1
        labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
        labels = labels.scatter_(1, label_mask, label_ids)
        # print(labels)
        # try:
        output = self.model(inputs_embeds=inputs_embeds.to(self.device),attention_mask=attention_mask.to(self.device).bool(),labels=labels.to(self.device),\
                            replace_embeds=self.replace_embeds,blocked_indices=self.blocked_indices,spell_length=self.spell_length)
        loss, logits = output.loss, output.logits
        # # print(logits)
        # print(logits.shape)
        pred_ids = torch.argsort(logits, dim=2, descending=True)
        # print(pred_ids.shape)
        # print(label_mask)
        hit1 = 0
        top10 = []
        pos_num = 0
        pos_num_all = 0
        pre_pos_num = 0
        for i in range(bz):
            top10.append([])
            pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
            top_50 = []
            for pred in pred_seq:
                # if pred in self.allowed_vocab_ids:
                top_50.append(pred)
                if (len(top_50) >= 10):
                    break
            pre = 0
            for id in top_50:
                if (id == 2748):
                    pre = 2748
                    break
                elif (id == 2053):
                    pre = 2053
                    break
            if (pre == label_ids[i, 0]):
                hit1 += 1
            if label_ids[i, 0] == 2748:
                pos_num_all += 1
            if pre == 2748:
                pre_pos_num += 1
            if label_ids[i, 0] == 2748 and pre == 2748:
                pos_num += 1
        if return_candidates:
            return loss, hit1, top10
        return loss, hit1, pos_num, pos_num_all, pre_pos_num
