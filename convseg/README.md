# CONV-SEG
Convolutional neural network for Chinese word segmentation (CWS). The corresponding paper [Convolutional Neural Network with Word Embeddings for Chinese Word Segmentation](https://arxiv.org/pdf/1711.04411.pdf) has been accepted
by IJCNLP2017.

[The original tensorflow 1.x implementation by the author](https://github.com/chqiwang/convseg)

## Dependencies
  * python 3.7
  * pytorch 1.5.0
## Data
Downlaod `data.zip` from [here](https://drive.google.com/file/d/0B-f0oKMQIe6sQVNxeE9JeUJfQ0k/view) (Note that the `SIGHAN` datasets should only be used for research purposes). Extract `data.zip` to this directory. So the file tree would be:
```
convseg
|	data
|	|	datasets
|	|	|	sighan2005-pku
|	|	|	|	train.txt
|	|	|	|	dev.txt
|	|	|	|	test.txt
|	|	|	sighan2005-msr
|	|	|	|	train.txt
|	|	|	|	dev.txt
|	|	|	|	test.txt
|	|	embeddings
|	|	|	news_tensite.w2v200
|	|	|	news_tensite.pku.words.w2v50
|	|	|	news_tensite.msr.words.w2v50
|	model.py
|	train.py
|	train_cws.sh
|	utils.py
|	cws.py
|	README.md
```
## How to use
modify the parameter in train_cws.sh and run `./train_cws.sh pku 0`

you can customize the `bash` file or add `perl` file for evaluation
## Innovations
implement the data iterator by imitating the class `torch.utils.data.DataLoader` in the `utils` file
