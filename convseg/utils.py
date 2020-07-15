# -*- ecoding: utf-8 -*-
# @ModuleName: utils
# @Function: 
# @Author: Yuxuan Xi
# @Time: 2020/7/9 16:39

from collections import defaultdict
import numpy as np

def create_dict(item_list,add_unk=False,add_pad=False):
    '''create a dictionary of items from a list of items'''
    assert type(item_list) in (list,tuple)
    dct=defaultdict(int)
    for items in item_list:
        for item in items:
            dct[item]+=1
    if add_unk:
        dct['<PAD>']=1e20#make sure the id of <PAD> is 0 in the function create_mapping
    if add_pad:
        dct['<UNK>']=1e10
    return dict(dct)

def create_mapping(items):
    '''
    create a mapping (item to id) or (id to item) from a dctionary
    :param items: dict or list
    :return: item2id,id2item
    '''
    if type(items) is dict:
        items=sorted(items.items(),key=lambda x:x[1],reverse=True)
        item2id={item[0]:id for id,item in enumerate(items)}
        id2item={id:item[0] for id,item in enumerate(items)}
        return dict(item2id),dict(id2item)
    if type(items) is list:
        item2id={item:id for id,item in enumerate(items)}
        id2item={id:item for id,item in enumerate(items)}
        return dict(item2id),dict(id2item)

def create_input(batch):
    '''
    deal with a batch,pad the sentences at the end in a batch to the same length
    :param batch:
    :return:
    '''
    lengths=[len(sent) for sent in batch[0]]
    maxlen=max(2,max(lengths))
    ret=[]
    for part in batch:
        pp=[]
        for sent,length in zip(part,lengths):
            assert len(sent)==length

            if length<maxlen:
                sent+=[0]*(maxlen-length)#do the padding
            pp.append(np.array(sent))
        ret.append(np.array(pp))
    ret.append(np.array(lengths))
    return ret

def data_to_ids(data,mappings):
    '''
    map text data to ids
    :param data:list of all different parts of data
    :param mappings:the mapping dict for different parts of data
    :return:the ids of the text data
    '''

    def strQ2B(ustring):#全角变半角
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:
                inside_code = 32
            elif 65281 <= inside_code <= 65374:
                inside_code -= 65248
            rstring += chr(inside_code)
        return rstring

    def strB2Q(ustring):#半角变全角
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 32:
                inside_code = 12288
            elif 32 <= inside_code <= 126:
                inside_code += 65248
            rstring += chr(inside_code)
        return rstring

    def map(string,mapping):
        if string in mapping:
            return mapping[string]
        string=strQ2B(string)
        if string in mapping:
            return mapping(string)
        string=strB2Q(string)
        if string in mapping:
            return mapping(string)
        return mapping['<UNK>']

    def map_seq(seq,mapping):
        return [map(string,mapping) for string in seq]

    ret=[]

    for part,mapping in zip(data,mappings):
        pp=[map_seq(seq,mapping) for seq in part]
        ret.append(pp)
    return ret

class data_loader(object):
    def __init__(self,dataset,batch_size=1,shuffle=False):
        assert len(dataset)>0
        assert all(len(item)==len(dataset[0]) for item in dataset)
        #make sure the length of data in the different parts is the same
        self.dataset=list(zip(*dataset))
        self.batch_size=batch_size
        self.shuffle=shuffle

    def __iter__(self):#shuffle the data for every epoch
        if self.shuffle:
            np.random.shuffle(self.dataset)
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.dataset)

class _DataLoaderIter(object):
    def __init__(self,loader):
        self.dataset=loader.dataset
        self.batch_size=loader.batch_size
        self.index=0
        self.data_num=len(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index>=self.data_num:
            raise StopIteration
        ret=zip(*self.dataset[self.index:min(self.index+self.batch_size,self.data_num)])
        ret=list(ret)
        self.index+=self.batch_size
        return create_input(ret)