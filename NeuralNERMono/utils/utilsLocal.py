from __future__ import print_function

__author__ = 'rudramurthy'

import os
import sys
import string
import io
import codecs
import numpy as np
import torch
from torch.autograd import Variable

def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8',errors='ignore') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])
        wv = np.loadtxt(wv)

    dictionary = dict()
    reverseDict = []
# dummy word for zero-padding
    dictionary["</SSSSSSSSSSSS>"] = len(dictionary)
    reverseDict.append("</SSSSSSSSSSSSS>")

    for everyWord in vocabulary:
        dictionary[everyWord.lower()] = len(dictionary)
        reverseDict.append(everyWord.lower())

    dictionary["</S>"] = len(dictionary)
    reverseDict.append("</S>")

    dimension = len(wv[0])
    vec = np.zeros(dimension)

    wordEmbedding = np.vstack( [vec, wv, vec])

    return wordEmbedding, dictionary, reverseDict, wordEmbedding.shape[0], dimension

def readCoNLL(filename, charDictionary, tagDictionary, charReverseDict, tagReverseDict, ner_tag_field):
    documents = []
    labels = []

    tagIndex = ner_tag_field

    sentences = []

    maxSequenceLength = 0

    with codecs.open(filename, 'r', encoding="utf8", errors='ignore') as fp:
        for line in fp:
            line = line.strip()
            if line:
                tokens = []
                tokens.append(line.split("\t")[0])
                tokens.append(line.split("\t")[ tagIndex ].upper())

                for everyChar in line.split("\t")[0]:
                    if everyChar not in charDictionary:
                        charDictionary[everyChar] = len(charDictionary)
                        charReverseDict.append(everyChar)

                if line.split("\t")[tagIndex].upper() not in tagDictionary:
                    tagDictionary[line.split("\t")[tagIndex].upper()] = len(tagDictionary)
                    tagReverseDict.append(line.split("\t")[tagIndex].upper())

                sentences.append(tokens)
            else:
                sentence = []
                for everyWord in sentences:
                    sentence.append(everyWord[0])

                documents.append(sentence)

                target = []
                for everyWord in sentences:
                    target.append(everyWord[1])

                labels.append(target)

                if maxSequenceLength < len(sentences):
                    maxSequenceLength = len(sentences)

                sentences = []

    return documents, labels, maxSequenceLength

def sortTrainData(trainData, trainTag):

    sentenceLengths = []
    for everySentence in trainData:
        sentenceLengths.append(len(everySentence))

    sentenceLengths = torch.Tensor(sentenceLengths)

    sorted_length, sorted_index = torch.sort(sentenceLengths, dim=0, descending=True)

    new_sentence_order = sorted_index.tolist()

    newTrainData = [trainData[i] for i in new_sentence_order]
    newTrainTag = [trainTag[i] for i in new_sentence_order]

    return newTrainData, newTrainTag

def batch(iterable1, iterable2, n=1):
    l = len(iterable1)
    for ndx in range(0, l, n):
        yield iterable1[ndx:min(ndx + n, l)], iterable2[ndx:min(ndx + n, l)]

def constructBatch(batchSentences, batchLabels, embedding_vocab, vocabularySize, tagVocabulary, charVocabulary, max_filter_width, use_gpu):
    transformedBatch = []

    batch_sequence_lengths = []
    max_sequence_length = 0
    batch_size = len(batchSentences)

    batch_actual_sum = 0

# for everySentence in the batch
    for i in range(len(batchSentences)):
        # if maximum sentence length is less than current sentence length
        if max_sequence_length < len(batchSentences[i]):
            max_sequence_length = len(batchSentences[i])
        # add current sentence length to batch_sequence_lengths list
        batch_sequence_lengths.append(len(batchSentences[i]))
        # total actual examples in the batch
        batch_actual_sum = batch_actual_sum + len(batchSentences[i])

    max_character_sequence_length = 0
    # for everySentence in the batch
    for i in range(len(batchSentences)):
        # for everyWord in the everySentence
        for j in range(len(batchSentences[i])):
            if len(batchSentences[i][j]) > max_character_sequence_length:
                max_character_sequence_length = len(batchSentences[i][j])

    if max_character_sequence_length < max_filter_width:
        max_character_sequence_length = max_filter_width

# the input to word embedding layer would be actual number of words
    wordInputFeature = torch.LongTensor(len(batchSentences), max_sequence_length).fill_(0)

    count = 0
    for i in range(len(batchSentences)):
        for j in range(len(batchSentences[i])):
            if batchSentences[i][j].lower() in embedding_vocab:
                wordInputFeature[i][j] = embedding_vocab[batchSentences[i][j].lower()]
            else:
                wordInputFeature[i][j] = embedding_vocab["</S>"]
            count = count + 1

# the input to subword feature extractor layer would be actual number of words
    charInputFeatures = torch.zeros(len(batchSentences) * max_sequence_length, 1, max_character_sequence_length * len(charVocabulary))

    count = 0
    for i in range(len(batchSentences)):
        for j in range(len(batchSentences[i])):

            temp = torch.zeros(max_character_sequence_length * len(charVocabulary))
            # pad it with a special start symbol
            temp[charVocabulary["<S>"]] = 1.0

            # for every character in the jth word
            for k in range(len(batchSentences[i][j])):
                if k < (max_character_sequence_length - 1):
                    if batchSentences[i][j][k] in charVocabulary:
                        temp[(k + 1) * len(charVocabulary) + charVocabulary[batchSentences[i][j][k]]] = 1.0

            # if number of characters is less than max_character_sequence_length
            if len(batchSentences[i][j]) < max_character_sequence_length:
                temp[ len(batchSentences[i][j])  * len(charVocabulary) + charVocabulary["</S>"]] = 1.0

            charInputFeatures[i * max_sequence_length + j][0] = temp
            count =  count + 1



# similarly construvt the output target labels, everySentence in every batch one-by-one
    batch_target = torch.LongTensor(len(batchSentences), max_sequence_length, ).fill_(0)
    batch_target_prev = torch.FloatTensor(len(batchSentences), max_sequence_length, len(tagVocabulary)).fill_(0)
    mask = torch.FloatTensor(len(batchSentences), max_sequence_length).fill_(0)
    index = 0

    for i in range(len(batchSentences)):
        for j in range(len(batchSentences[i])):
            batch_target[i][j] = tagVocabulary[batchLabels[i][j]]

            if j != 0:
                batch_target_prev[i][j][tagVocabulary[batchLabels[i][j-1]]] = 1.0
            mask[i][j] = 1.0
            index = index + 1



    batch_input = []
    if use_gpu == 1:
        batch_input.append(Variable(wordInputFeature.cuda()))
        batch_input.append(Variable(charInputFeatures.cuda()))
    else:
        batch_input.append(Variable(wordInputFeature))
        batch_input.append(Variable(charInputFeatures))

    if use_gpu == 1:
        return batch_input, Variable(torch.LongTensor(batch_sequence_lengths)), batch_size, max_sequence_length, Variable(batch_target.cuda()), Variable(mask.cuda()), Variable(batch_target_prev.cuda())
    else:
        return batch_input, Variable(torch.LongTensor(batch_sequence_lengths)), batch_size, max_sequence_length, Variable(batch_target), Variable(mask), Variable(batch_target_prev)
