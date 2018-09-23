import progressbar
import six
import datetime

import numpy as np
import chainer.functions as F
from chainer.backends import cuda
import re

UNK = 0
EOS = 1
PAD = -1
#seqenceの長さのMaxとMin
minlen = 1
maxlen = 50

def word_ids(voPath):
    #wordをidにするword_ids
    #辞書{文字:数字} 文字を入れたら数字が得られる
    with open(voPath) as f:
        # +2 for UNK and EOS
        #line.strip 改行空白除去
        words_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    words_ids['<UNK>'] = 0
    words_ids['<EOS>'] = 1
    words_ids['<PAD>'] = -1 #いる？


    return words_ids

def ids_word(word_ids):
    #idをwordにする
    #辞書{数字:文字} 数字を入れたら文字が得られる
    ids_words = {i: w for w, i in word_ids.items()}

    return ids_words

def load_data(vocabulary, peVocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    wordData = []
    #humanData = []
    count = 0
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            parson = re.match(r'[^(:|\n)]*:', line)#:を抜いた人の名前2単語以上ならどうしよ？
            line = re.sub(r'[^(:|\n)]*:', "", line)
            if parson != None:
                parson = parson.group(0)[:-1].strip()
            else:
                count += 1
                parson = "none"
            words = line.strip().split()
            #numpyにせんとメモリが
            wordArray = np.array([vocabulary.get(w, UNK) for w in words], dtype=np.int32)
            parson = np.array(peVocabulary.get(parson, UNK), dtype=np.int32)
            wordData.append((parson, wordArray))
    print(count)
    return wordData

def count_lines(path):
    #lineを数える
    with open(path) as f:
        return sum([1 for _ in f])

def sequence_embed(embed, xs):
    #各文の長さ len(xs)はバッチサイズ
    if len(np.shape(xs)) != 1:
        x_len = [len(x) for x in xs]
        #cumsum指定された軸方向に足し合わされていった値を要素とする配列が返されます。
        #x_len[:-1]は一番最後を抜いたリスト
        x_section =np.cumsum(x_len[:-1])
        #xsはVariableかnumpy.ndarrayかcupy.ndarrayのタプル
        #concatは配列の結合axis=0は縦に結合行数が多くなる
        ex = embed(F.concat(xs, axis=0))
        #spritされてタプルが帰る結合したのを元に戻してそう
        exs = F.split_axis(ex, x_section, 0)
    else:
        exs = embed(xs)
    return exs

def makeData(inPath, outPath, voPath, peVoPath):
    #return [(source wordID0, target wordID0),(),()], {word:ID,..}, $0の分割版いる？
    vocabulary = word_ids(voPath)
    peVocabulary = word_ids(peVoPath)
    #[np.array0, np.array1,...] ID
    #inData[(parson, wordArray), (), ()]
    inData = load_data(vocabulary, peVocabulary, inPath)
    outData = load_data(vocabulary, peVocabulary, outPath)
    assert len(inData) == len(outData)#ちょい厳しい
    train_data = [#write
            ((s[0], np.append(s[1], EOS)), (t[0], np.append(t[1], EOS)))
            for s, t in six.moves.zip(inData, outData)
            if (minlen <= len(s[1])+1 <= maxlen
                and
                minlen <= len(t[1])+1 <= maxlen)
        ]
    #おまけ
    print('[{}] Dataset loaded.'.format(datetime.datetime.now()))
    train_source_unknown = calculate_unknown_ratio(
            [s[1] for s, _ in train_data])
    train_target_unknown = calculate_unknown_ratio(
            [t[1] for _, t in train_data])

    print('vocabulary size: %d' % len(vocabulary))
    print('persona vocabulary size %d' % len(peVocabulary))
    print('Train data size: %d' % len(train_data))
    print('source unknown ratio: %.2f%%' % (
        train_source_unknown * 100))
    print('target unknown ratio: %.2f%%' % (
        train_target_unknown * 100))

    #train_data[((inparson, inArray),(outparson, outArray)), ((inparson, inArray),(outparson, outArray))]
    return (train_data, vocabulary, peVocabulary, inData, outData)

def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


