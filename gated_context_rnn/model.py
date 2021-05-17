import copy
import math
import numpy as np
from numpy.testing._private.utils import requires_memory
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module): # attention module
    def __init__(self, hidden_size, num_heads, dropout_rate=0.3):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_hidden_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_heads * self.head_hidden_size

        self.fc_q = nn.Linear(self.hidden_size, self.all_head_size)
        self.fc_k = nn.Linear(self.hidden_size, self.all_head_size)
        self.fc_v = nn.Linear(self.hidden_size, self.all_head_size)

        self.fc_o = nn.Linear(self.all_head_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.shape[1]
        mixed_Q = self.fc_q(hidden_state)
        mixed_K = self.fc_k(hidden_state)
        mixed_V = self.fc_v(hidden_state)

        # head 개수 만큼 query, key, value를 잘라 줌
        # permute하는 이유 => head별로 연산을 해야하기 때문에 head를 sequence length보다 앞으로 위치시킴
        Q = mixed_Q.view(batch_size, -1, self.num_heads, self.head_hidden_size).permute(0, 2, 1, 3) 
        K = mixed_K.view(batch_size, -1, self.num_heads, self.head_hidden_size).permute(0, 2, 1, 3) 
        V = mixed_V.view(batch_size, -1, self.num_heads, self.head_hidden_size).permute(0, 2, 1, 3)

        #scaled-dot product
        attention_score = torch.matmul(Q, K.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.head_hidden_size)

        if attention_mask is not None:
            attention_score = attention_score.masked_fill(attention_score==0, -np.inf)

        attention_weight = self.softmax(attention_score)
        attention_weight = self.dropout(attention_weight) # 각 토큰 간의 연관성이 scalar로 표현됨 => dimension: (batch, num_heads, sequence_length, sequence_length)
        
        context_vector = torch.matmul(attention_weight, V) # dimension: (batch, num_heads, sequence_length, head_hidden_size)
        context_vector = context_vector.permute(0,2,1,3)
        context_vector = context_vector.reshape(batch_size, -1, self.all_head_size)

        attention_output = self.fc_o(context_vector)

        return attention_output

class PositionWiseFeedForward(nn.Module): # position-wise feed forward
    def __init__(self, hidden_size, pwff_size=400, dropout_rate=0.3):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, pwff_size)
        self.fc2 = nn.Linear(pwff_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropuout = nn.Dropout(dropout_rate)

    def forward(self, x):
        f1_out = self.fc1(x)
        activation_output = self.relu(f1_out)
        f2_out = self.fc2(activation_output)
        
        pwff_output = self.dropuout(f2_out)

        return pwff_output

class Block(nn.Module): # encoder block : self-attenton, add&normalize, position-wise feed forward, add&normalize
    def __init__(self, hidden_size=100, num_heads=4, pwpf_size=400, dropout_rate=0.3):
        super(Block, self).__init__()
        self.attn_layer = Attention(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate)
        self.attn_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.pwff_layer = PositionWiseFeedForward(hidden_size=hidden_size, pwff_size=pwpf_size, dropout_rate=dropout_rate)
        self.pwff_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        attn_input = x
        attn_output = self.attn_layer(attn_input, attention_mask=attention_mask).permute(1,0,2)
        attn_output = attn_output + attn_input #residual connection
        normalized_attn_output = self.attn_norm(attn_output)
        
        pwff_input = normalized_attn_output
        pwff_output = self.pwff_layer(pwff_input)
        pwff_output = pwff_output + pwff_input #residual connection
        normalized_pwff_output = self.pwff_norm(pwff_output)

        return normalized_pwff_output



class TransformerEncoder(nn.Module): # transformer encoder
    def __init__(self, num_layer = 6, hidden_size=100, num_heads=4, pwpf_size=400, dropout_rate=0.3):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList()

        for _ in range(num_layer):
            layer = Block(hidden_size=hidden_size, num_heads=num_heads, pwpf_size=pwpf_size, dropout_rate=dropout_rate)
            self.layer.append(copy.deepcopy(layer)) 

    def forward(self, x, attention_mask=None):
        all_layer_output = []

        for layer in self.layer:
            layer_output = layer(x, attention_mask)
            all_layer_output.append(layer_output)

        return all_layer_output

class ContextEncoder(nn.Module):
    def __init__(self, device, seq_length, input_size=100, hidden_size=100):
        super(ContextEncoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.transformer = TransformerEncoder()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)

    def position_encoding_init(self, n_position, d_pos_vec):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
            if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)
        
    def forward(self, x, umask=None, post=False):
        batch_size = x.shape[1]
        umask_ = umask.unsqueeze(-1)
        umask_ = umask_.expand(umask.shape[0], batch_size, 100)
        pos = self.position_encoding_init(x.shape[0], x.shape[2]).to(self.device)
        pos = pos.unsqueeze(0).repeat(batch_size, 1, 1).permute(1,0,2)
        pos = pos * umask_
        x = x + pos
        transformer_outputs = self.transformer(x, umask)
        residual_outputs = transformer_outputs[-1]

        hidden_states, _ = self.lstm(residual_outputs)
        hidden_states = hidden_states

        if post:
            context = hidden_states[0]
        else:
            context = hidden_states[-1]

        return context

class Model(nn.Module):
    def __init__(self, device, hidden_size, num_label, max_length=100, dropout_rate=0.3):
        super(Model, self).__init__()
        self.device = device
        self.context_encoder1 = ContextEncoder(device=device, seq_length=max_length)
        self.context_encoder2 = ContextEncoder(device=device, seq_length=max_length)
        self.context_encoder3 = ContextEncoder(device=device, seq_length=max_length)
        self.context_encoder4 = ContextEncoder(device=device, seq_length=max_length)

        self.speaker_embedding = nn.Linear(2, 100)

        self.gate1 = nn.Linear(hidden_size * 2, 100)
        self.gate2 = nn.Linear(hidden_size * 2, 100)
        self.gate3 = nn.Linear(hidden_size * 2, 100)
        self.gate4 = nn.Linear(hidden_size * 2, 100)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_1 = nn.Linear(300, num_label)
        self.softmax = nn.Softmax(dim=0)

    def data_preprocess_batch(self, utterances, idx, umask, speaker):
        batch_size = utterances.shape[1]
        ones = torch.ones(umask.shape[0], dtype=torch.int32, device=self.device)
        one = torch.ones(1, dtype=torch.int32, device=self.device)

        self_u = torch.zeros(0).type(utterances.type()).to(self.device) # batch, D_e
        inter_u = torch.zeros(0).type(utterances.type()).to(self.device) # batch, D_e
        self_speakers = torch.zeros(0).type(umask.type()).to(self.device) # batch, D_e
        inter_speakers = torch.zeros(0).type(umask.type()).to(self.device) # batch, D_e

        speaker = speaker.permute(1,0).to(self.device)
        utterances = utterances.permute(1,0,2)
        umask = umask.permute(1,0)
        
        for i in range(0, batch_size):
            target_utterance = utterances[i][idx].to(self.device)

            if speaker[i][idx] == 0:
                inter_speaker = speaker[i].clone()
                inter_speaker[idx] = inter_speaker[idx] + 1
                self_speaker = ones - speaker[i]
                self_speaker = self_speaker.unsqueeze(1)
                inter_speaker = inter_speaker.unsqueeze(1)
                self_utterances = utterances[i] * self_speaker
                inter_utterances = utterances[i] * inter_speaker

            else:
                self_speaker = speaker[i]
                inter_speaker = ones - speaker[i]
                inter_speaker[idx] = inter_speaker[idx] + 1
                self_speaker = self_speaker.unsqueeze(1)
                inter_speaker = inter_speaker.unsqueeze(1)
                self_utterances = utterances[i] * self_speaker
                inter_utterances = utterances[i] * inter_speaker

            umask_ = umask[i].unsqueeze(1)
            self_utterances = self_utterances * umask_
            inter_utterances = inter_utterances * umask_

            self_utterances = self_utterances.unsqueeze(1)
            inter_utterances = inter_utterances.unsqueeze(1)

            self_u = torch.cat([self_u, self_utterances], 1)
            inter_u = torch.cat([inter_u, inter_utterances], 1)
            self_speakers = torch.cat([self_speakers, self_speaker], 1)
            inter_speakers = torch.cat([inter_speakers, inter_speaker], 1)

        utterances = utterances.permute(1,0,2)
        umask = umask.permute(1,0)

        return self_u[:idx+1], self_u[idx:], inter_u[:idx+1], self_speakers[:idx+1], self_speakers[idx:], inter_speakers[:idx+1], utterances[:idx+1], umask[:idx+1]

    def data_preprocess(self, utterances, idx, umask, speaker):
        batch_size = utterances.shape[1]
        ones = torch.ones(umask.shape, dtype=torch.int32, device=self.device)
        speaker = speaker.to(self.device)
        target_utterance = utterances[idx].to(self.device)

        utterances = utterances.permute(1,0,2).to(self.device)

        if speaker[idx] == 0:
            inter_speaker = speaker
            speaker = ones - speaker
            self_utterances = utterances * speaker
            inter_utterances = utterances * inter_speaker

        else:
            speaker = speaker
            inter_speaker = ones - speaker
            self_utterances = utterances * speaker
            inter_utterances = utterances * inter_speaker
        
        self_utterances = self_utterances * umask
        inter_utterances = inter_utterances * umask

        self_utterances = self_utterances.permute(1,0,2)
        inter_utterances = inter_utterances.permute(1,0,2)
        utterances = utterances.permute(1,0,2)
        inter_utterances[idx] = target_utterance

        inter_speaker[idx] = 1

        return self_utterances[:idx+1], self_utterances[idx:], inter_utterances[:idx+1], speaker[:idx+1], speaker[idx:], inter_speaker[:idx+1], utterances[:idx+1], umask[:idx+1]

    def data_preprocess2(self, utterances, idx, umask):
        return utterances[:idx+1], utterances[idx:], utterances[idx], umask[:idx+1], umask[idx:]
    
    def forward(self, U, umask, speaker, s_embedding):
        sentence_length = umask.shape[0]
        s_embedding = s_embedding.to(self.device)
        s_embedded = self.speaker_embedding(s_embedding)
        s_embedded = s_embedded.to(self.device)
        utterances = U.to(self.device) + s_embedded.to(self.device)
        logits = torch.zeros(0).type(U.type()) # batch, D_e
        logits = logits.to(self.device)
        umask = umask.to(self.device)

        for idx in range(sentence_length):
            #pre_utterances, post_utterances, target_utterances, pre_umask, post_umask = self.data_preprocess2(utterances=utterances, idx=idx, umask=umask)
            pre_self_utterances, post_self_utterances,pre_inter_utterances, pre_self_umask, post_self_umask, pre_inter_umask, pre_utterances, pre_mask = self.data_preprocess_batch(utterances=utterances, idx=idx, umask=umask, speaker=speaker)
            pre_self_utterances=pre_self_utterances.to(self.device)
            post_self_utterances = post_self_utterances.to(self.device)
            pre_inter_utterances=pre_inter_utterances.to(self.device)

            pre_self_context = self.context_encoder1(pre_self_utterances, pre_self_umask)
            post_self_context = self.context_encoder1(post_self_utterances, post_self_umask, True)
            pre_inter_context = self.context_encoder1(pre_inter_utterances, pre_inter_umask)
            pre_global_context = self.context_encoder1(pre_utterances, pre_mask)

            pre_self_context = self.gate1(pre_self_context)
            post_self_context = self.gate1(post_self_context)
            pre_inter_context = self.gate1(pre_inter_context)
            pre_global_context = self.gate1(pre_global_context)

            inertia_vector = torch.cat([pre_self_context, post_self_context, pre_inter_context], -1)
            
            context = F.tanh(inertia_vector)
            context = self.dropout(context)
            context = self.fc_1(context)
            logit =F.log_softmax(context, -1)
            logits = torch.cat([logits, logit], 0)

        return logits

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss
