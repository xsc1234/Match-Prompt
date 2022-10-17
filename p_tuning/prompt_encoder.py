import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        # self.cloze_mask = [
        #     [1] * self.cloze_length  # first cloze
        # ]
        self.cloze_mask_q1 = [
            [1] * self.cloze_length[0]
        ]
        self.cloze_mask_q2 = [
            [1] * self.cloze_length[1]
        ]
        self.cloze_mask_prompt = [
            [1] * self.cloze_length[2]
        ]
        self.cloze_mask_q1 = torch.LongTensor(self.cloze_mask_q1).bool().to(self.device)
        self.cloze_mask_q2 = torch.LongTensor(self.cloze_mask_q2).bool().to(self.device)
        self.cloze_mask_prompt = torch.LongTensor(self.cloze_mask_prompt).bool().to(self.device)
        # print(self.cloze_mask_q1)
        # print(self.cloze_mask_q2)
        # print(self.cloze_mask_prompt)
        #[[1, 1, 1, 1, 1]]
        # print(self.cloze_mask.shape)
        self.q1_len = self.cloze_length[0]
        self.q2_len = self.cloze_length[0]+self.cloze_length[1]
        self.prompt_len = self.cloze_length[0]+self.cloze_length[1]+self.cloze_length[2]

        self.seq_indices_q1 = torch.LongTensor(list(range(self.q1_len))).to(self.device)
        self.seq_indices_q2 = torch.LongTensor(list(range(self.q1_len,self.q2_len))).to(self.device)
        self.seq_indices_prompt = torch.LongTensor(list(range(self.q2_len,self.prompt_len))).to(self.device)
        # print(self.seq_indices_q1)
        # print(self.seq_indices_q2)
        # print(self.seq_indices_prompt)
        #tensor([0, 1, 2, 3, 4], device='cuda:0')
        # print(self.seq_indices.shape)
        # embedding
        self.embedding = torch.nn.Embedding(self.prompt_len, self.hidden_size).to(self.device)
        # self.embedding = torch.nn.Embedding(len(self.cloze_mask_q2[0]), self.hidden_size).to(self.device)
        # self.embedding = torch.nn.Embedding(len(self.cloze_mask_q2[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head1 = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True).to(self.device)
        self.lstm_head2 = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True).to(self.device)
        self.lstm_head3 = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True).to(self.device)
        self.mlp_head1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)
        self.mlp_head2 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)
        self.mlp_head3 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)
        print("init prompt encoder...")

    def forward(self):
        input_embeds_q1 = self.embedding(self.seq_indices_q1).unsqueeze(0).to(self.device)
        input_embeds_q2 = self.embedding(self.seq_indices_q2).unsqueeze(0).to(self.device)
        input_embeds_prompt = self.embedding(self.seq_indices_prompt).unsqueeze(0).to(self.device)
        # print(input_embeds_q1.shape)
        # print(input_embeds_q2.shape)
        # print(input_embeds_prompt.shape)
        # tensor([[[1.4838, 0.3263, 0.0266, ..., -1.2007, -0.7030, 0.7687],
        #          [0.3393, 1.1916, -1.4759, ..., -0.2123, 0.0091, -0.2394],
        #          [1.3532, 0.0073, 0.4631, ..., 1.5452, 0.4061, 1.0228],
        #          [0.7332, 0.2699, -1.2401, ..., -1.6983, 0.8545, -1.1657],
        #          [0.1746, -1.4495, -0.3195, ..., -0.1280, -1.1282, 0.6752]]],
        #print(input_embeds.shape)
        # print(self.seq_indices)
        output_embeds_q1 = self.mlp_head1(self.lstm_head1(input_embeds_q1)[0]).squeeze().to(self.device)
        # output_embeds_q1 = output_embeds_q1.unsqueeze(0)
        # print(output_embeds_q1.shape)
        #torch.Size([3, 768])
        output_embeds_q2 = self.mlp_head2(self.lstm_head2(input_embeds_q2)[0]).squeeze().to(self.device)
        # output_embeds_q2 = output_embeds_q2.unsqueeze(0)
        #print(output_embeds_q2.shape)
        output_embeds_prompt = self.mlp_head3(self.lstm_head3(input_embeds_prompt)[0]).squeeze().to(self.device)
        #print(output_embeds_prompt.shape)
        output_embeds = torch.cat([output_embeds_q1,output_embeds_q2,output_embeds_prompt],dim=0)
        # # print(output_embeds)
        # print(output_embeds.shape)
        # torch.Size([11, 768])

        return output_embeds


class PromptEncoder_with_input(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device,args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding

        self.lstm_head1 = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True).to(self.device)
        self.lstm_head2 = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True).to(self.device)
        self.lstm_head3 = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True).to(self.device)
        self.mlp_head1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)
        self.mlp_head2 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)
        self.mlp_head3 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)
        print("init prompt encoder...")

    def forward(self,s1,s2,q):
        input_embeds_s1 = s1.float()
        input_embeds_s2 = s2.float()
        input_embeds_q = q.float()
        #print(input_embeds_s1)
        # print(input_embeds_s1.shape)
        # print(input_embeds_s2.shape)
        # print(input_embeds_q.shape)
        # torch.Size([16, 46, 768])
        # torch.Size([16, 42, 768])
        # torch.Size([16, 7, 768])
        output_embeds_q1 = self.mlp_head1(self.lstm_head1(input_embeds_s1)[0]).squeeze().to(self.device)
        output_embeds_q1 = output_embeds_q1[:,0:6,:]
        # print(output_embeds_q1.shape)
        # output_embeds_q1 = output_embeds_q1.unsqueeze(0)
        #print(output_embeds_q1.shape)
        output_embeds_q2 = self.mlp_head2(self.lstm_head2(input_embeds_s2)[0]).squeeze().to(self.device)
        output_embeds_q2 = output_embeds_q2[:,0:6,:]
        # print(output_embeds_q2.shape)
        # output_embeds_q2 = output_embeds_q2.unsqueeze(0)
        #print(output_embeds_q2.shape)
        output_embeds_prompt = self.mlp_head3(self.lstm_head3(input_embeds_q)[0]).squeeze().to(self.device)
        output_embeds_prompt = output_embeds_prompt[:,0:7,:]
        # print(output_embeds_prompt.shape)
        #print(output_embeds_prompt.shape)
        output_embeds = torch.cat([output_embeds_q1,output_embeds_q2,output_embeds_prompt],dim=1)
        # print(output_embeds)
        # print(output_embeds.shape)
        # torch.Size([16, 6, 768])
        # torch.Size([16, 6, 768])
        # torch.Size([16, 7, 768])
        # torch.Size([16, 19, 768])

        return output_embeds


class PromptEncoder_all_layer(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        # self.cloze_mask = [
        #     [1] * self.cloze_length  # first cloze
        # ]
        self.cloze_mask_q1 = [
            [1] * self.cloze_length[0]
        ]
        self.cloze_mask_q2 = [
            [1] * self.cloze_length[1]
        ]
        self.cloze_mask_q1 = torch.LongTensor(self.cloze_mask_q1).bool().to(self.device)
        self.cloze_mask_q2 = torch.LongTensor(self.cloze_mask_q2).bool().to(self.device)
        # print(self.cloze_mask_q1)
        # print(self.cloze_mask_q2)
        # print(self.cloze_mask_prompt)
        #[[1, 1, 1, 1, 1]]
        # print(self.cloze_mask.shape)
        self.q1_len = self.cloze_length[0]
        self.q2_len = self.cloze_length[0]+self.cloze_length[1]

        self.seq_indices_q1 = torch.LongTensor(list(range(self.q1_len))).to(self.device)
        self.seq_indices_q2 = torch.LongTensor(list(range(self.q1_len,self.q2_len))).to(self.device)
        # print(self.seq_indices_q1)
        # print(self.seq_indices_q2)
        # print(self.seq_indices_prompt)
        #tensor([0, 1, 2, 3, 4], device='cuda:0')
        # print(self.seq_indices.shape)
        # embedding
        self.embedding = torch.nn.Embedding(self.q2_len, self.hidden_size).to(self.device)
        # self.embedding = torch.nn.Embedding(len(self.cloze_mask_q2[0]), self.hidden_size).to(self.device)
        # self.embedding = torch.nn.Embedding(len(self.cloze_mask_q2[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head1 = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True).to(self.device)
        self.lstm_head2 = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True).to(self.device)

        self.mlp_head1 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)
        self.mlp_head2 = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)

        print("init prompt encoder...")

    def forward(self):
        input_embeds_q1 = self.embedding(self.seq_indices_q1).unsqueeze(0).to(self.device)
        input_embeds_q2 = self.embedding(self.seq_indices_q2).unsqueeze(0).to(self.device)
        # print(input_embeds_q1.shape)
        # print(input_embeds_q2.shape)
        # print(input_embeds_prompt.shape)
        # tensor([[[1.4838, 0.3263, 0.0266, ..., -1.2007, -0.7030, 0.7687],
        #          [0.3393, 1.1916, -1.4759, ..., -0.2123, 0.0091, -0.2394],
        #          [1.3532, 0.0073, 0.4631, ..., 1.5452, 0.4061, 1.0228],
        #          [0.7332, 0.2699, -1.2401, ..., -1.6983, 0.8545, -1.1657],
        #          [0.1746, -1.4495, -0.3195, ..., -0.1280, -1.1282, 0.6752]]],
        #print(input_embeds.shape)
        # print(self.seq_indices)
        output_embeds_q1 = self.mlp_head1(self.lstm_head1(input_embeds_q1)[0]).squeeze().to(self.device)
        # output_embeds_q1 = output_embeds_q1.unsqueeze(0)
        #torch.Size([3, 768])
        output_embeds_q2 = self.mlp_head2(self.lstm_head2(input_embeds_q2)[0]).squeeze().to(self.device)
        # output_embeds_q2 = output_embeds_q2.unsqueeze(0)
        #print(output_embeds_q2.shape)
        #print(output_embeds_prompt.shape)
        output_embeds = torch.cat([output_embeds_q1,output_embeds_q2],dim=0)
        # torch.Size([11, 768])

        return output_embeds