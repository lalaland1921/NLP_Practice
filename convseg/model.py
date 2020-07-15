# -*- ecoding: utf-8 -*-
# @ModuleName: model
# @Function: 
# @Author: Yuxuan Xi
# @Time: 2020/7/10 8:38

import torch
import torch.nn as nn
from utils import *
from torch.nn.utils import weight_norm
from torch.nn.functional import relu,sigmoid,tanh
from torchcrf import CRF
from utils import data_loader
import os
from copy import deepcopy

class ConvSeg(nn.Module):
    def __init__(self,vocab_size,emb_size,word_vocab_size,word_emb_size,word_window_size,
                 hidden_layers,kernel_size,channels,
                 num_tags, use_crf, lamd, dropout_emb,
                 dropout_hidden,  use_bn, use_wn, active_type
                 ):
        super(ConvSeg, self).__init__()
        self.word_window_size=word_window_size
        self.hidden_layers=hidden_layers
        #self.kernel_size=kernel_size
        #self.num_tags=num_tags
        self.embedding=nn.Embedding(vocab_size,emb_size)
        self.word_embedding=nn.Embedding(word_vocab_size,word_emb_size)
        self.active_type=active_type
        self.lamb=lamd
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dropout_hidden:
            self.dropout_hidden=nn.Dropout(dropout_hidden)
        else:self.dropout_hidden=None
        if dropout_emb:
            self.drop_embed=nn.Dropout(dropout_emb)
        else:self.drop_embed=None
        self.conv_layers=[]
        pre_channels=emb_size+word_window_size*word_emb_size
        for i in range(hidden_layers):
            cur_channels=channels[i]
            convs=[[nn.Conv1d(pre_channels,cur_channels,kernel_size,padding=kernel_size//2,bias=True)],
                   [nn.Conv1d(pre_channels,cur_channels,kernel_size,padding=kernel_size//2,bias=True)]]
            if use_wn:
                convs=[list(map(weight_norm,conv)) for conv in convs]
            if use_bn:
                convs=[conv+[nn.BatchNorm1d(cur_channels)] for conv in convs]
            self.conv_layers.append(convs)
            pre_channels=cur_channels

        self.fc=nn.Linear(pre_channels,num_tags)
        if use_crf:
            self.crf=CRF(num_tags)
        else:self.crf=None

    def forward(self,x,seq_lengths):

        items,words=x[0],x[1:]
        mask=torch.range(0,seq_lengths.max()-1).unsqueeze(0)\
            .type_as(seq_lengths).expand(len(seq_lengths),seq_lengths.max()).\
            lt(seq_lengths.unsqueeze(1)).type(torch.float)
        item_embeddings=self.embedding(items)
        word_embeddings=[]
        for i in range(self.word_window_size):
            word_embeddings.append(self.word_embedding(words[i]))
        embedding_output=torch.cat([item_embeddings]+word_embeddings,2)
        if self.drop_embed:
            embedding_output=self.drop_embed(embedding_output)

        hidden_output=embedding_output.permute(0,2,1)#(batch_size,channel,seq_length)
        for layer in range(self.hidden_layers):
            w=v=hidden_output
            for module in self.conv_layers[layer][0]:
                w=module(w)
            for module in self.conv_layers[layer][1]:
                v=module(v)
            if self.active_type=='glu':
                hidden_output=w*sigmoid(v)
            elif self.active_type=='relu':
                hidden_output=relu(w)
            elif self.active_type=='gtu':
                hidden_output=tanh(w)*sigmoid(v)
            elif self.active_type=='tanh':
                hidden_output=tanh(w)
            elif self.active_type=='linear':
                hidden_output=w
            elif self.active_type=='bilinear':
                hidden_output=w*v
            #mask paddings
            hidden_output=hidden_output*mask.unsqueeze(1).expand(-1,hidden_output.size(1),-1)
            #dropout on hidden output
            if self.dropout_hidden:
                hidden_output=self.dropout_hidden(hidden_output)
        scores=self.fc(hidden_output.permute(0,2,1))
        #(bathch_size,seq_length,hidden_embedding)->(batch_size,seq_length,tag_num)
            #the score of padding positions is zero,also the tag id is zero,the loss is reduced
        return scores

    def train(self,train_data, dev_data, test_data,epochs,batch_size,optimizer,model_path):
        bias_list = (param for name, param in self.named_parameters() if name[-4:] == 'bias')
        other_list = (param for name, param in self.named_parameters() if name[-4:] != 'bias')
        parameters = [{'params': bias_list, 'weight_decay': 0},
                      {'params': other_list,'weight_decay':self.lamb}]
        # no l2 normalization for bias
        # Parse optimization method and parameters.
        optimizer = optimizer.split('_')
        optimizer_name = optimizer[0]
        optimizer_options = [eval(i) for i in optimizer[1:]]
        optimizer={
            'sgd':torch.optim.SGD,
            'adadelta':torch.optim.Adadelta,
            'adam':torch.optim.Adam,
            #add more
        }[optimizer_name](parameters,*optimizer_options)



        DataLoader=data_loader(train_data,batch_size,shuffle=True)
        total_batch_num=(len(DataLoader)-1)//batch_size+1
        best_accuracy=0
        for epoch in range(epochs):
            for i,batch in enumerate(DataLoader):
                x,seq_length,tags=np.array(batch[:-2]),batch[-1],batch[-2]
                x=torch.from_numpy(x).long().to(self.device)
                seq_length=torch.from_numpy(seq_length).long().to(self.device)
                tags=torch.from_numpy(tags).long().to(self.device)
                scores=self.forward(x,seq_length)#(batch_size,seq_length,num_tags)
                if self.crf:
                    emissions=scores.permute(1,0,2)#(seq_length,batch_size,num_tags)
                    tags=tags.permute(1,0)#(seq_length,batch_size)
                    mask = torch.range(0, seq_length.max() - 1).unsqueeze(0) \
                        .type_as(seq_length).expand(len(seq_length), seq_length.max()). \
                        lt(seq_length.unsqueeze(1)).type(torch.float).permute(1,0)
                    #(seq_length,batch_size)
                    loss = self.crf(emissions, tags,mask.bool())
                else:
                    criterion=nn.CrossEntropyLoss()
                    loss=criterion(scores.reshape(-1,scores.size(-1)),tags.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i%10==0:
                    print("epoch {}/{} ,step {}/{},the loss: {}".format(epoch,epochs,i,total_batch_num,loss.item()))
                    new_accuracy = self.evaluation(dev_data, batch_size)
                    if new_accuracy > best_accuracy:
                        best_accuracy = new_accuracy
                        #self.best_model = deepcopy(self.modules())
                        torch.save(self.state_dict(), model_path)
                        print("model saved in {}, the accuracy: {}".format(model_path, best_accuracy))
            print("epoch {} finished".format(epoch))
            new_accuracy=self.evaluation(dev_data,batch_size)
            if new_accuracy>best_accuracy:
                best_accuracy=new_accuracy
                #self.best_model = deepcopy(self.model)
                torch.save(self.state_dict(),model_path)
                print("model saved in {}, the accuracy: {}".format(model_path,best_accuracy))

        self.load_state_dict(torch.load(self.model_path))# best model
        test_accuracy=self.evaluation(test_data)
        print("the training has finished, the accuracy in test data is {}".format(test_accuracy))

    def predict(self,test_data):#list of array
        x,seq_length=np.array(test_data[:-1]), test_data[-1]
        x = torch.from_numpy(x).long().to(self.device)
        seq_length = torch.from_numpy(seq_length).long().to(self.device)
        scores = self.forward(x, seq_length)  # (batch_size,seq_length,num_tags)
        if self.crf:
            emissions=scores.premute(1,0,2)
            predicts=self.crf.decode(emissions)
        else:
            _, predicts = torch.max(scores.data, 2)
        return predicts

    def evaluation(self,test_data,batch_size):
        DataLoader = data_loader(test_data, batch_size, shuffle=True)
        # the max_length must be determined in a batch,
        # to make the least influence of abnormal data, just do it in a batch?
        # or this is the normal operation?
        total=correct=0
        with torch.no_grad():
            for batch in DataLoader:
                x, tags = batch[:-2]+[batch[-1]], batch[-2]
                predicts=self.predict(x)
                tags = torch.from_numpy(tags)

                total += tags.size(0) * tags.size(1)
                correct += (predicts == tags).sum().item()
        return 1.0 * correct / total

    def inference(self,test_data_ids,batch_size,id2item,id2tag,result_path):
        result_text=''
        DataLoader = data_loader(test_data_ids, batch_size, shuffle=True)
        with torch.no_grad():
            for batch in DataLoader:
                x, tags = batch[:-2]+[batch[-1]], batch[-2]
                predicts=self.predict(x).numpy()
                items=x[0]
                for i,sentence in enumerate(items):
                    for j,item in enumerate(sentence):
                        result_text+=id2item[item]
                        if id2tag[predicts[i][j]]=='E':
                            result_text+=' '
                    result_text+='\n'
        print("the inference is done!")
        with open(result_path,'w',encoding='utf-8') as f:
            f.write(result_text)



