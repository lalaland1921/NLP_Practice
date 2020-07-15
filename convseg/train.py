# -*- ecoding: utf-8 -*-
# @ModuleName: train
# @Function: 
# @Author: Yuxuan Xi
# @Time: 2020/7/11 15:35

import sys,codecs,os,pickle
from argparse import ArgumentParser
from model import ConvSeg
from utils import *
import torch

'''class FlushFile:
    """
    A wrapper for File, allowing users see result immediately.
    """
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        #self.f.flush()
        '''

if __name__ == '__main__':
    #sys.stdout = FlushFile(sys.stdout)

    parser = ArgumentParser()
    parser.add_argument('--task', dest='task',default='cws')
    parser.add_argument('--training_path', dest='training_path', default='data/datasets/sighan2005-pku/train.txt')
    parser.add_argument('--dev_path', dest='dev_path', default='data/datasets/sighan2005-pku/dev.txt')
    parser.add_argument('--test_path', dest='test_path', default='data/datasets/sighan2005-pku/test.txt')
    parser.add_argument('--pre_trained_emb_path', dest='pre_trained_emb_path', default=None)
    parser.add_argument('--pre_trained_word_emb_path', dest='pre_trained_word_emb_path', default=None)
    parser.add_argument('--model_root', dest='model_root', default='model-pku')
    parser.add_argument('--emb_size', dest='emb_size', type=int, default=200)
    parser.add_argument('--word_window', dest='word_window', type=int, default=4)
    parser.add_argument('--hidden_layers', dest='hidden_layers', type=int, default=5)
    parser.add_argument('--channels', dest='channels', type=list, default=[100]*5)
    parser.add_argument('--kernel_size', dest='kernel_size', type=int, default=3)
    parser.add_argument('--word_emb_size', dest='word_emb_size', type=int, default=50)
    parser.add_argument('--use_bn', dest='use_bn', type=int, default=0)
    parser.add_argument('--use_wn', dest='use_wn', type=int, default=1)
    parser.add_argument('--dropout_emb', dest='dropout_emb', type=float, default=0.2)
    parser.add_argument('--dropout_hidden', dest='dropout_hidden', type=float, default=0.2)
    parser.add_argument('--active_type', dest='active_type', default='glu')
    parser.add_argument('--lamd', dest='lamd', type=float, default=0)
    parser.add_argument('--fix_word_emb', dest='fix_word_emb', type=int, default=0)
    parser.add_argument('--reserve_all_word_emb', dest='reserve_all_word_emb', type=int, default=0)
    parser.add_argument('--use_crf', dest='use_crf', type=int, default=0)
    parser.add_argument('--optimizer', dest='optimizer', default='adam_0.001')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=1000)
    parser.add_argument('--max_epoches', dest='max_epoches', type=int, default=100)
    parser.add_argument('--use_inference',dest='use_inference',type=int,default=1)

    args = parser.parse_args()
    print(args)

    TASK = __import__(args.task)

    train_data, dev_data, test_data = (
    TASK.read_data_file(codecs.open(args.training_path, 'r', 'utf8'), word_window=args.word_window),
    TASK.read_data_file(codecs.open(args.dev_path, 'r', 'utf8'), word_window=args.word_window),
    TASK.read_data_file(codecs.open(args.test_path, 'r', 'utf8'), word_window=args.word_window))

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    mappings_path = os.path.join(args.model_root, 'mappings.pkl')
    parameters_path = os.path.join(args.model_root, 'parameters.pkl')
    data_id_path=os.path.join(args.model_root,'data.pkl')
    model_path=os.path.join(args.model_root,'model.ckpt')
    result_path=os.path.join(args.model_root,'result.txt')#save the inference result of test data

    # Load or create mappings.
    if os.path.isfile(mappings_path):
        item2id, id2item, tag2id, id2tag, word2id, id2word = pickle.load(open(mappings_path, 'rb'))
    else:
        item2id, id2item = create_mapping(create_dict(train_data[0], add_unk=True, add_pad=True))
        #word2id,id2word=create_mapping(create_dict(train_data[1:-1],add_unk=True,add_pad=True))
        tag2id, id2tag = create_mapping(create_dict(train_data[-1]))
        words=[]
        for t in train_data[1:-1]:
            words.extend(t)
        for t in dev_data[1:-1]:
            words.extend(t)
        for t in test_data[1:-1]:
            words.extend(t)
        word2id,id2word=create_mapping(create_dict(words,add_pad=True,add_unk=True))
        pickle.dump((item2id, id2item, tag2id, id2tag, word2id, id2word), open(mappings_path, 'wb'))

    word_window_size = len(train_data) - 2
    parameters = {
        'vocab_size': len(item2id),
        'emb_size': args.emb_size,
        'word_window_size': word_window_size,
        'word_vocab_size': len(word2id),
        'word_emb_size': args.word_emb_size,
        'hidden_layers': args.hidden_layers,
        'channels': args.channels,
        'kernel_size': args.kernel_size,
        'use_bn': args.use_bn,
        'use_wn': args.use_wn,
        'num_tags': len(tag2id),
        'use_crf': args.use_crf,
        'lamd': args.lamd,
        'dropout_emb': args.dropout_emb,
        'dropout_hidden': args.dropout_hidden,
        'active_type': args.active_type
    }

    if os.path.isfile(parameters_path):
        parameters_old = pickle.load(open(parameters_path, 'rb'))
        if parameters != parameters_old:
            raise Exception('Network parameters are not consistent!')
    else:
        pickle.dump(parameters, open(parameters_path, 'wb'))

    if os.path.isfile(data_id_path):
        train_data_ids,dev_data_ids,test_data_ids=pickle.load(open(data_id_path,'rb'))
    else:
        # Convert data to corresponding ids.
        train_data_ids = data_to_ids(
            train_data, [item2id] + [word2id] * word_window_size + [tag2id]
        )
        dev_data_ids = data_to_ids(
            dev_data, [item2id] + [word2id] * word_window_size + [tag2id]
        )
        test_data_ids = data_to_ids(
            test_data, [item2id] + [word2id] * word_window_size + [tag2id]
        )
        pickle.dump((train_data_ids,dev_data_ids,test_data_ids),open(data_id_path,'wb'))

    print('data to id finished.')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ConvSeg(**parameters)
    model.to(device)
    if os.path.isfile(model_path): # continue the training from where it is interrupted
        model.load_state_dict(torch.load(model_path))
    
    model.train(train_data=train_data_ids, dev_data=dev_data_ids, test_data=test_data_ids,
                    epochs=args.max_epoches, batch_size=args.batch_size,
                    optimizer=args.optimizer, model_path=model_path)


    if args.use_inference:
        model.inference(test_data_ids,args.eval_batch_size,id2item,id2tag,result_path)

'''
test data_loader
'''
'''import utils
dataloader=data_loader(dev_data_ids,100)
for i,batch in enumerate(dataloader):
    print(i)
    '''
