import loadData
import personaModel

import argparse
from nltk.translate import bleu_score
import chainer
from chainer import training
from chainer.training import extensions
import numpy as np
from chainer.backends import cuda
import chainer.functions as F
import random

UNK = 0
EOS = 1
PAD = -1

def main():
    parser = argparse.ArgumentParser(description='Chainer persona test')
    parser.add_argument('VOCAB', help='vocabulary file')
    parser.add_argument('PERSONA_VOCAB', help='persona vocabulary file')
    parser.add_argument('--batch', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='number of units')#defalt=1024
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--use', '-use', default='null', 
                        help='use made model file path')
    args = parser.parse_args()

    vocabulary = loadData.word_ids(args.VOCAB)
    peVocabulary = loadData.word_ids(args.PERSONA_VOCAB)
    model = personaModel.Model(vocabulary, peVocabulary, args.unit, args.batch, args.gpu)
    chainer.serializers.load_npz(args.out+"/"+args.use)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
        xp = cuda.cupy
    else:
        xp = np

    print("input persona")
    persona = input()
    persona = model.personaEmbed(persona)
    

    while(True):
        print("input sentence")
        inputSentence = input()
        words = inputSentence.strip.split()
        idsArray = xp.asarray([vocabulary.get(w, UNK) for w in words], dtype=xp.int32)
        idsArray = F.concat((idsArray, EOS), axis=1)
        result = model.predict(idsArray, persona)
        result = ' '.join([vocabulary.get(w, UNK) for w in result[0]])
        print("output")
        print(result)


