# Targeted Sentiment Analysis with Advanced Span-based Boundary Detection 

This repo contains the code and data of the following paper:

In this paper, we design a novel span-based framework for the task of targeted sentiment analysis., which is shown in image/:


This framework consists of two components:  
- Aspect extraction  
- Sentiment prediction

Both of two components utilize [BERT](https://github.com/huggingface/pytorch-pretrained-BERT) as backbone network. 

## Requirements
- Python 3
- [Pytorch 1.1](https://pytorch.org/) 

Download the uncased [BERT-Base](https://drive.google.com/file/d/13I0Gj7v8lYhW5Hwmp5kxm3CTlzWZuok2/view?usp=sharing) model and unzip it in the current directory. 



## How to run the model
You can  try to run the joint model, since the parameters of each data-set have been set:

```
python run.py 
```

And the result are replaced in out/


