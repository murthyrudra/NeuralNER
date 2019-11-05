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
	dictionary = dict()
	reverseDict = []
# dummy word for zero-padding
	dictionary["</SSSSSSSSSSSS>"] = len(dictionary)
	reverseDict.append("</SSSSSSSSSSSSS>")

	wv = []
	dimension = 0
	with codecs.open(file_name, 'r', 'utf-8',errors='ignore') as f_in:
		for line in f_in:
			line = line.strip()
			# print(line)
			if line:
				vocabulary = line.split(' ')[0]
				if vocabulary.lower() not in dictionary:
					temp = []
					dictionary[vocabulary.lower()] = len(dictionary)
					reverseDict.append(vocabulary.lower())

					if dimension == 0:
						dimension = len(line.split(' ')[1:])

					if dimension != len(line.split(' ')[1:]):
						print(line)
						print(str(dimension) +"\t" + str(len(line.split(' ')[1:])))
						exit()
					for i in line.split(' ')[1:]:
						temp.append(float(i))

					wv.append(temp)

	wv_np = np.array(wv)

	dictionary["<unk>"] = len(dictionary)
	reverseDict.append("<unk>")

	vec = np.zeros(dimension)

	wordEmbedding = np.vstack( [vec, wv_np, vec])

	return wordEmbedding, dictionary, reverseDict, wordEmbedding.shape[0], dimension

def readCoNLL(filename, charDictionary, tagDictionary, ner_tag_field, embedd_dict):
	documents = []
	labels = []

	tagIndex = ner_tag_field
	wordIndex = 0

	sentences = []

	maxSequenceLength = 0
	totalWords = 0
	unknownWords = 0

	with codecs.open(filename, 'r', encoding="utf8", errors='ignore') as fp:
		for line in fp:
			line = line.strip()
			if line:
				# line = line.lower()
				if not line.startswith("#"):
					tokens = []
					tokens.append(line.split("\t")[wordIndex])
					tokens.append(line.split("\t")[ tagIndex ].upper())

					if not tagDictionary.get_freeze():
						tagDictionary.add(line.split("\t")[tagIndex].upper())

					if not charDictionary.get_freeze():
						for everyChar in line.split("\t")[wordIndex]:
							charDictionary.add(everyChar)

					sentences.append(tokens)
			else:
				sentence = []
				target = []
				if len(sentences) > 0:
					for i in range(len(sentences)):
						if sentences[i][0].endswith('।'):
							if sentences[i][1] == "O":
								sentence.append(sentences[i][0])
								target.append(sentences[i][1])

								documents.append(sentence)
								labels.append(target)

								if maxSequenceLength < len(sentence):
									maxSequenceLength = len(sentence)

								sentence = []
								target = []
							else:
								sentence.append(sentences[i][0])
								target.append(sentences[i][1])
						else:
							sentence.append(sentences[i][0])
							target.append(sentences[i][1])

							totalWords = totalWords + 1
							if sentences[i][0] not in embedd_dict:
								unknownWords = unknownWords + 1
					if len(sentence) > 0:
						documents.append(sentence)
						labels.append(target)

						if maxSequenceLength < len(sentence):
							maxSequenceLength = len(sentence)

					sentences = []

	print("Total words = " + str(totalWords))
	print("Unknown words = " + str(unknownWords))
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

def constructBatch(batchSentences, batchLabels, inputVocabulary, targetVocabulary, charVocabulary, max_filter_width, use_gpu):
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
			wordInputFeature[i][j] = inputVocabulary.__get_word_train__(batchSentences[i][j])
			count = count + 1

# the input to subword feature extractor layer would be actual number of words
	charInputFeatures = torch.zeros(len(batchSentences) * max_sequence_length, 1, max_character_sequence_length * charVocabulary.__len__())

	count = 0
	#for every setence in the batch
	for i in range(len(batchSentences)):
		#for every word in that sentence
		for j in range(len(batchSentences[i])):

			temp = torch.zeros(max_character_sequence_length  * charVocabulary.__len__())
			# pad it with a special start symbol
			ind = charVocabulary.__get_word__("<S>")
			temp[ind] = 1.0

			# for every character in the jth word
			for k in range(len(batchSentences[i][j])):
				if k < (max_character_sequence_length - 1):
					ind = charVocabulary.__get_word__(batchSentences[i][j][k])

					if ind !=  None:
						temp[(k + 1) * charVocabulary.__len__() + ind] = 1.0

			# if number of characters is less than max_character_sequence_length
			if len(batchSentences[i][j]) < max_character_sequence_length:
				ind = charVocabulary.__get_word__("</S>")
				temp[ len(batchSentences[i][j])  * charVocabulary.__len__() + ind] = 1.0

			charInputFeatures[i * max_sequence_length + j][0] = temp
			count =  count + 1

# similarly construvt the output target labels, everySentence in every batch one-by-one
	batch_target = torch.LongTensor(len(batchSentences), max_sequence_length, ).fill_(0)
	batch_target_prev = torch.FloatTensor(len(batchSentences), max_sequence_length, targetVocabulary.__len__()).fill_(0)
	mask = torch.FloatTensor(len(batchSentences), max_sequence_length).fill_(0)
	index = 0

	for i in range(len(batchSentences)):
		for j in range(len(batchSentences[i])):
			batch_target[i][j] = targetVocabulary.__get_word_train__(batchLabels[i][j])

			if j != 0:
				batch_target_prev[i][j][ targetVocabulary.__get_word_train__(batchLabels[i][j-1]) ] = 1.0
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
