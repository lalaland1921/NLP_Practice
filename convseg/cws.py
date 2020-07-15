# -*- ecoding: utf-8 -*-
# @ModuleName: cws
# @Function: 
# @Author: Yuxuan Xi
# @Time: 2020/7/9 15:37

import os,codecs

def process_sentence(sentence,bigram,word_window):
    words=sentence.strip().split()
    chars=[]
    ret=[]
    tags=[]
    for word in words:
        chars += list(word)
        if len(word)==1:
            tags.append('S')
        else:
            tags+=['B']+['M']*(len(word)-2)+['E']
    if bigram:
        chars=[' ',' ']+chars+[' ',' ']
        ret.append([a+b if a and b else '' for a,b in zip(chars[:-4],chars[1:-3])])#zip会自动取最短的
        ret.append([a+b if a and b else ' ' for a,b in zip(chars[1:-3],chars[2:-2])])
        ret.append([a+b if a and b else ' ' for a,b in zip(chars[2:-2],chars[3:-1])])
        ret.append([a+b if a and b else ' ' for a,b in zip(chars[3:-1],chars[4:])])
    elif word_window and word_window>=1:
        assert word_window<=4#in the paper the max of word window is 4
        ret.append(chars)
        if word_window>=2:
            chars=[' ',' ',' ']+chars+[' ',' ',' ']
            ret.append([a+b if a and b else '' for a,b in zip(chars[2:-4],chars[3:])])
            ret.append([a+b if a and b else '' for a,b in zip(chars[3:-3],chars[4:])])
        if word_window>=3:
            ret.append([a+b+c if a and b and c else '' for a,b,c in zip(chars[1:-5],chars[2:-4],chars[3:])])
            ret.append([a+b+c if a and b and c else '' for a,b,c in zip(chars[2:-4],chars[3:-3],chars[4:])])
            ret.append([a+b+c if a and b and c else '' for a,b,c in zip(chars[3:-3],chars[4:-2],chars[5:])])
        if word_window>=4:
            ret.append([a+b+c+d if a and b and c and d else '' for a,b,c,d in zip(chars[:-6],chars[1:-5],chars[2:-4],chars[3:])])
            ret.append([a+b+c+d if a and b and c and d else '' for a,b,c,d in zip(chars[1:-5],chars[2:-4],chars[3:-3],chars[4:])])
            ret.append([a+b+c+d if a and b and c and d else '' for a,b,c,d in zip(chars[2:-4],chars[3:-3],chars[4:-2],chars[5:])])
            ret.append([a+b+c+d if a and b and c and d else '' for a,b,c,d in zip(chars[3:-3],chars[4:-2],chars[5:-1],chars[6:])])

    ret.append(tags)
    return ret

def read_data_file(f,bigram=False,word_window=4):
    ret=[]
    for sentence in f:
        ret.append(process_sentence(sentence,bigram,word_window))
    return list(zip(*ret))
