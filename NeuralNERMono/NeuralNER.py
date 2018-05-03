from __future__ import print_function

__author__ = 'rudramurthy'
"""
Implementation of Neural NER.
"""

import os
import sys
import codecs

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid

import numpy as np
import torch
from torch.optim import Adam, SGD
from models.modules import BiCNNLSTMTranstion
from utils.utilsLocal import *
from torch.utils.data import DataLoader

uid = uuid.uuid4().hex[:6]

# The following function is copied from the Neuro NLP code made available at https://github.com/XuezheMax/NeuroNLP
def evaluate(output_file, save_dir):
	score_file = "%s/score_%s" % (save_dir, str(uid))
	os.system("eval/conll03eval.v2 < %s > %s" % (output_file, score_file))
	with open(score_file, 'r') as fin:
		fin.readline()
		line = fin.readline()
		fields = line.split(";")
		acc = float(fields[0].split(":")[1].strip()[:-1])
		precision = float(fields[1].split(":")[1].strip()[:-1])
		recall = float(fields[2].split(":")[1].strip()[:-1])
		f1 = float(fields[3].split(":")[1].strip())
	return acc, precision, recall, f1


def main():
	parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CNN')
	parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
	parser.add_argument('--batch_size', type=int, default=5, help='Number of sentences in each batch')
	parser.add_argument('--hidden_size', type=int, default=200, help='Number of hidden units in RNN')
	parser.add_argument('--num_filters', type=int, default=15, help='Number of filters in CNN')
	parser.add_argument('--min_filter_width', type=int, default=1, help='Number of filters in CNN')
	parser.add_argument('--max_filter_width', type=int, default=7, help='Number of filters in CNN')

	parser.add_argument('--learning_rate', type=float, default=0.4, help='Learning rate')
	parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
	parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
	parser.add_argument('--schedule', type=int, default=1, help='schedule for learning rate decay')
	parser.add_argument('--embedding_dict', help='path for embedding dict')
	parser.add_argument('--train')
	parser.add_argument('--dev')
	parser.add_argument('--test')

	parser.add_argument('--ner_tag_field', type=int, default=1, help='ner tag field')
	parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')

	parser.add_argument('--save-dir')
	parser.add_argument('--perform_evaluation', type=bool, default=False, help='perform evaluation only')

	args = parser.parse_args()

	train_path = args.train
	dev_path = args.dev
	test_path = args.test

	num_epochs = args.num_epochs
	batch_size = args.batch_size
	hidden_size = args.hidden_size

	num_filters = args.num_filters
	min_filter_width = args.min_filter_width
	max_filter_width = args.max_filter_width

	learning_rate = args.learning_rate
	momentum = 0.01 * learning_rate
	decay_rate = args.decay_rate
	gamma = args.gamma
	schedule = args.schedule

	embedding_path = args.embedding_dict

	save_dir = args.save_dir

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	evaluation = args.perform_evaluation

	embedd_dict, embedding_vocab, reverse_word_vocab, vocabularySize, embeddingDimension = load_embeddings(embedding_path)

	print("Read Word Embedding of dimension " + str(embeddingDimension) + " for " + str(vocabularySize) + " number of words")

	charVocabulary = dict()
	charReverseVocabulary = []

	charVocabulary["<S>"] = len(charVocabulary)
	charReverseVocabulary.append("<S>")
	charVocabulary["</S>"] = len(charVocabulary)
	charReverseVocabulary.append("</S>")

	tagVocabulary = dict()
	tagReverseVocabulary = []

	trainCorpus, trainLabelsRaw, maxTrainLength = readCoNLL(train_path, charVocabulary, tagVocabulary, charReverseVocabulary, tagReverseVocabulary, args.ner_tag_field)
	print("Train Corpus contains " + str(len(trainCorpus)) + " sentences and maximum sentence length is " + str(maxTrainLength))

	trainCorpusRawSorted, trainLabelsRawSorted = sortTrainData(trainCorpus, trainLabelsRaw)
	print("Sorted the train Corpus based on length ")

	devCorpus, devLabelsRaw, maxDevLength = readCoNLL(dev_path, charVocabulary, tagVocabulary, charReverseVocabulary, tagReverseVocabulary, args.ner_tag_field)
	testCorpus, testLabelsRaw, maxTestLength = readCoNLL(test_path, charVocabulary, tagVocabulary, charReverseVocabulary, tagReverseVocabulary, args.ner_tag_field)

	print("Dev Corpus contains " + str(len(devCorpus)) + " sentences and maximum sentence length is " + str(maxDevLength))
	print("Test Corpus contains " + str(len(testCorpus)) + " sentences and maximum sentence length is " + str(maxTestLength))

	print("Read " + str(len(charVocabulary)) + " number of characters")

	print(tagVocabulary)

	print("Number of epochs = " +  str(num_epochs))
	print("Mini-Batch size = " +  str(batch_size))
	print("Bi-LSTM Hidden size = " +  str(hidden_size))
	print("Features per CNN filter = " +  str(num_filters))
	print("Minimum ngrams for CNN filter = " +  str(min_filter_width))
	print("Maximum ngrams for CNN filter = " +  str(max_filter_width))
	print("Initial Learning Rate = " +  str(learning_rate))

	use_gpu = torch.cuda.is_available()

	network = BiCNNLSTMTranstion(vocabularySize, embeddingDimension, min_filter_width, max_filter_width, len(charVocabulary), num_filters, hidden_size, len(tagVocabulary) , embedd_dict)

	# network = BiLSTM(vocabularySize, embeddingDimension, hidden_size, len(tagVocabulary) , embedd_dict)

	lr = learning_rate

	optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

	num_batches = len(trainCorpus) / batch_size + 1

	dev_f1 = 0.0
	dev_acc = 0.0
	dev_precision = 0.0
	dev_recall = 0.0
	test_f1 = 0.0
	test_acc = 0.0
	test_precision = 0.0
	test_recall = 0.0
	best_epoch = 0

	print("Training....")

	if args.use_gpu == 1:
		network.cuda()

	if evaluation:
		network.load_state_dict(torch.load(save_dir + "/model"))

		network.eval()
		tmp_filename = '%s/%s_test%d' % (save_dir, str(uid), epoch)

		with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
			for inputs, labels in batch(testCorpus, testLabelsRaw, 1):
				x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, embedding_vocab, vocabularySize, tagVocabulary, charVocabulary, max_filter_width)

				loss, preds = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev)

				count = 0

				for i in range(len(inputs)):
					for j in range(len(inputs[i])):
						writer.write(inputs[i][j] + " " + labels[i][j] + " " + tagReverseVocabulary[preds[i][j]])
						writer.write("\n")
					writer.write("\n")

			writer.close()

		acc, precision, recall, f1 = evaluate(tmp_filename)
		print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))
	else:

		prev_error = 1000.0

		for epoch in range(1, num_epochs + 1):
			print('Epoch %d ( learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (epoch, lr, decay_rate, schedule))

			train_err = 0.
			train_corr = 0.
			train_total = 0.

			start_time = time.time()
			num_back = 0
			network.train()

			count = 0
			count_batch = 0

			for inputs, labels in batch(trainCorpusRawSorted, trainLabelsRawSorted, batch_size):

				x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, embedding_vocab, vocabularySize, tagVocabulary, charVocabulary, max_filter_width, args.use_gpu)

				optim.zero_grad()
				# loss = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask)

				loss, _ = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

				loss.backward()
				optim.step()

				train_err += loss.data[0]
				train_total += batch_length.data.sum()

				count = count + current_batch_size
				count_batch = count_batch + 1

				time_ave = (time.time() - start_time) / count
				time_left = (num_batches - count_batch) * time_ave

			print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / count, time.time() - start_time))

			network.eval()
			tmp_filename = '%s/%s_dev%d' % (save_dir, str(uid), epoch)
			current_epoch_loss = 0.0

			with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
				for inputs, labels in batch(devCorpus, devLabelsRaw, 1):
					x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, embedding_vocab, vocabularySize, tagVocabulary, charVocabulary, max_filter_width, args.use_gpu)

					loss, _ = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)
					current_epoch_loss = current_epoch_loss + loss.data[0]

					loss, preds = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

					count = 0

					for i in range(len(inputs)):
						for j in range(len(inputs[i])):
							writer.write(inputs[i][j] + " " + labels[i][j] + " " + tagReverseVocabulary[preds[i][j]])
							writer.write("\n")
							writer.write("\n")

				writer.close()

			acc, precision, recall, f1 = evaluate(tmp_filename, save_dir)
			print('dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))

			if epoch > 1:
				if current_epoch_loss > prev_error:
					lr = lr * 0.7
					optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

					network.load_state_dict(torch.load(save_dir + "/model"))
					network.eval()

					if lr < 0.002:
						network.eval()
						tmp_filename = '%s/%s_test%d' % (save_dir, str(uid), epoch)

						with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
							for inputs, labels in batch(testCorpus, testLabelsRaw, 1):
								x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, embedding_vocab, vocabularySize, tagVocabulary, charVocabulary, max_filter_width, args.use_gpu)

								loss, preds = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

								count = 0

								for i in range(len(inputs)):
									for j in range(len(inputs[i])):
										writer.write(inputs[i][j] + " " + labels[i][j] + " " + tagReverseVocabulary[preds[i][j]])
										writer.write("\n")
									writer.write("\n")

							writer.close()

						acc, precision, recall, f1 = evaluate(tmp_filename, save_dir)
						print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))

						exit()
				else:
					prev_error = current_epoch_loss
					torch.save(network.state_dict(), save_dir + "/model")

					network.eval()
					tmp_filename = '%s/%s_test%d' % (save_dir, str(uid), epoch)

					with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
						for inputs, labels in batch(testCorpus, testLabelsRaw, 1):
							x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, embedding_vocab, vocabularySize, tagVocabulary, charVocabulary, max_filter_width, args.use_gpu)

							loss, preds = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

							count = 0

							for i in range(len(inputs)):
								for j in range(len(inputs[i])):
									writer.write(inputs[i][j] + " " + labels[i][j] + " " + tagReverseVocabulary[preds[i][j]])
									writer.write("\n")
								writer.write("\n")

						writer.close()

					acc, precision, recall, f1 = evaluate(tmp_filename, save_dir)
					print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))
			else:
				prev_error = current_epoch_loss
				torch.save(network.state_dict(), save_dir + "/model")

			# if epoch % 5 == 0:
			# 	network.eval()
			# 	tmp_filename = '%s/%s_test%d' % (save_dir, str(uid), epoch)
			#
			# 	with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
			# 		for inputs, labels in batch(testCorpus, testLabelsRaw, 1):
			# 			x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, embedding_vocab, vocabularySize, tagVocabulary, charVocabulary, max_filter_width, args.use_gpu)
			#
			# 			loss, preds = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)
			#
			# 			count = 0
			#
			# 			for i in range(len(inputs)):
			# 				for j in range(len(inputs[i])):
			# 					writer.write(inputs[i][j] + " " + labels[i][j] + " " + tagReverseVocabulary[preds[i][j]])
			# 					writer.write("\n")
			# 				writer.write("\n")
			#
			# 		writer.close()
			#
			# 	acc, precision, recall, f1 = evaluate(tmp_filename, save_dir)
			# 	print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))
			#



if __name__ == '__main__':
	main()
