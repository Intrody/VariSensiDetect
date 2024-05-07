# VariSensiDetect
基于错误检测的含变体字敏感词检测方法


1. 敏感词检测方法主要通过BERT对输入句子进行错误检测，根据检测结果对疑似词进行敏感匹配。其主要流程图例存放与src/图解.vsdx
2. 含变体字的敏感词句子部分数据集位于data/sensitive.csv，完整数据集可以经过申请获取、并且只能用于研究。
3. 汉字匹配代码入口为src/similar/siacnnatten.py，数据集存放于data/train.csv,validation.csv,test.csv
