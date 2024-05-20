# coding:utf-8
import torch
import string
import re
from transformers import BertTokenizer, BertForMaskedLM

# 加载中文BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('Chinesebert')
model = BertForMaskedLM.from_pretrained('Chinesebert')

import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
import numpy as np


def is_punctuation(char):
    # 判断是否是英文标点
    if char in string.punctuation:
        return True

    # 判断是否是中文标点
    # 中文标点的Unicode范围为：\u3000-\u303F 和 \uFF00-\uFFEF
    if '\u3000' <= char <= '\u303F' or '\uFF00' <= char <= '\uFFEF' or char == ' ':
        return True

    return False


def check_word_at_position(sentence, segments, position):
    char = sentence[position]
    for segment in segments:
        if char in segment and len(segment) > 1 and (sentence[position:position + len(segment)] == segment or sentence[position - len(segment) + 1:position + 1] == segment):
            #print(f"位置 {position} 上的字 '{char}' 被组成新词: {segment}")
            return 1, segment
    #print(f"位置 {position} 上的字 '{char}' 没有被组成新词。")
    return 0, None


def bert_dectect_batch(sentences):
    # 将所有句子tokenize并获得token的位置
    tokenized_sentences = [tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sent))) for sent in sentences]
    token_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_sentences]


    for i, (tokens, ids) in enumerate(zip(tokenized_sentences, token_ids)):
        #print(f"Sentence {i + 1}: {tokens}")
        scores = []
        for j in range(1, len(ids) - 1):
            input_ids = ids.copy()
            # 将当前位置的token替换为[MASK]
            input_ids[j] = tokenizer.mask_token_id

            # 转换为PyTorch张量
            input_tensor = torch.tensor([input_ids])

            # 获取预测结果
            with torch.no_grad():
                predictions = model(input_tensor)[0]




            #print(f"Processing token {j} out of {len(ids) - 2}")
            try:
                original_token_id = tokenizer.convert_tokens_to_ids([tokens[j]])[0]
                original_probability = predictions[0, j, original_token_id].item()
            except IndexError:
                print("IndexError: list index out of range")
                print("Tokens:", tokens)
                print("Length of tokens:", len(tokens))
                print("Index j:", j)
                continue



            if tokens[j] == '[UNK]':
                tokens[j] = sentences[i][j - 1]
            print(f"位置 {j}: 原始词 '{tokens[j]}', 原始字概率: {original_probability}")

            scores.append(original_probability)

        tokens = tokens[1:-1]

        error_word  = detect_words_in_sentence(sentences[i],scores,tokens)
        print(error_word)
        with open("sensitive/detectoutput_66.tsv", "a",encoding='utf-8')as output_file:
            label = 1 if error_word else 0
            output_file.write(f"{sentences[i]}\t{label}\n")




def split_sentences(text):
    # 使用正则表达式匹配标点符号
    # 注意：这里的标点符号模式可能需要根据实际需求进行调整
    pattern = r'[,.!?，。！？;\n]'
    # 分割句子
    sentences = re.split(pattern, text)
    # 去除空白句子
    return [sent.strip() for sent in sentences if sent.strip()]

def detect_outliers_mad(array):
    median = np.median(array)
    mad = np.median(np.abs(array - median))
    threshold = 1.5 * mad
    outliers = [value for value in array if np.abs(value - median) > threshold]
    return outliers

def detect_words_in_sentence(sentence,scores,tokens):
    scoreAvg = sum(scores) / len(scores)
    tokhan = tok(sentence)
    errwords = []
    # median = np.median(scores)
    # mad = np.median(np.abs(scores - median))
    # threshold = 4 * mad
    for i in range(len(scores)):
        if scores[i] < scoreAvg * 0.4 and not is_punctuation(tokens[i ]):
            errwords.append([tokens[i ], i])




    print(sentence)
    if not errwords:
        print('未检测到疑似词')
    return errwords


if __name__ == '__main__':
    # 读取TSV文件第一列数据
    with open("sensitive/testdataset.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()

    sentences = [line.split('\t')[0].strip() for line in lines]
    batch_size = 8  # 设置批处理大小

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        bert_dectect_batch(batch_sentences)

