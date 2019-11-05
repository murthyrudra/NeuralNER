from __future__ import print_function
# Parts of the code are taken from NeuroNLP2 https://github.com/XuezheMax/NeuroNLP2/
__author__ = 'rudramurthy'
"""
Implementation of Bi-directional LSTM-CNNs model for NER.
"""

import os
import sys
import codecs

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import fnmatch
from utils.vocab import Vocab, CharVocab
import json
from tqdm import tqdm
import numpy as np
import math
import torch
from torch.optim import Adam, SGD
from models.modules import BiCNNLSTMTranstion
from utils.utilsLocal import *
from torch.nn.parameter import Parameter


def evaluate(output_file, save_dir):
	score_file = "%s/score_" % (save_dir)
	os.system("perl eval/conll03eval.v2 < %s > %s" % (output_file, score_file))
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
	parser = argparse.ArgumentParser(description='Training a Sequence Labeler with bi-directional LSTM-CNN')
	parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
	parser.add_argument('--batch_size', type=int, default=5, help='Number of sentences in each batch')
	parser.add_argument('--hidden_size', type=int, default=200, help='Number of hidden units in RNN')
	parser.add_argument('--num_filters', type=int, default=35, help='Number of filters in CNN')
	parser.add_argument('--min_filter_width', type=int, default=3, help='Number of filters in CNN')
	parser.add_argument('--max_filter_width', type=int, default=7, help='Number of filters in CNN')
	parser.add_argument('--embedDimension', type=int, default=300, help='embedding dimension')

	parser.add_argument('--learning_rate', type=float, default=0.4, help='Learning rate')
	parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
	parser.add_argument('--embedding_vectors', help='path for embedding dict')
	parser.add_argument('--embedding_dict_new', help='path for embedding dict')
	parser.add_argument('--train')
	parser.add_argument('--dev')
	parser.add_argument('--test')

	parser.add_argument('--vocabChar')
	parser.add_argument('--vocabOutput')
	parser.add_argument('--vocabInput')

	parser.add_argument('--ner_tag_field', type=int, default=1, help='ner tag field')
	parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')

	parser.add_argument('--fineTune', type=bool, default=False, help='fineTune pretrained word embeddings')

	parser.add_argument('--save-dir')
	parser.add_argument('--perform_evaluation', type=bool, default=False, help='perform evaluation only')
	parser.add_argument('--deploy', type=bool, default=False, help='deploy')

	parser.add_argument('--train_from', type=str, default="")

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
	gamma = args.gamma

	embedding_path = args.embedding_vectors

	save_dir = args.save_dir

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	evaluation = args.perform_evaluation

	inputVocabulary = Vocab()
	charVocabulary = CharVocab()
	targetVocabulary = Vocab()

	if args.vocabChar:
		with open(args.vocabChar, "r") as f:
			charVocabulary.__dict__ = json.load(f)
		charVocabulary.set_freeze()
		charVocabulary.process()

	if args.vocabOutput:
		with open(args.vocabOutput, "r") as f:
			targetVocabulary.__dict__ = json.load(f)
		targetVocabulary.set_freeze()
		targetVocabulary.process()

	embedding_vocab = None

	if args.embedding_vectors:
		print(args.embedding_vectors)
		embedd_dict, embedding_vocab, reverse_word_vocab, vocabularySize, embeddingDimension = load_embeddings(embedding_path)
		print("Read Word Embedding of dimension " + str(embeddingDimension) + " for " + str(vocabularySize) + " number of words")

		for everyWord in embedding_vocab:
			inputVocabulary.add(everyWord)
		inputVocabulary.set_freeze()
		inputVocabulary.process()
	else:
		if args.vocabInput:
			with open(args.vocabInput, "r") as f:
				inputVocabulary.__dict__ = json.load(f)
			inputVocabulary.set_freeze()
			inputVocabulary.process()
		else:
			print("Neither pre-trained word embeddings nor input vocabulary is specified")
			exit()

	if charVocabulary.__is_empty__():
		charVocabulary.add("<S>")
		charVocabulary.add("</S>")

	if evaluation:
		if not args.deploy:
			testCorpus, testLabelsRaw, maxTestLength = readCoNLL(test_path, charVocabulary, targetVocabulary, args.ner_tag_field, inputVocabulary)
			print("Test Corpus contains " + str(len(testCorpus)) + " sentences and maximum sentence length is " + str(maxTestLength))
			print("Read " + str(len(charVocabulary)) + " number of characters")

		else:
			testCorpus, maxTestLength = readUnlabeledData(test_path)
			print("Test Corpus contains " + str(len(testCorpus)) + " sentences and maximum sentence length is " + str(maxTestLength))
	else:
		trainCorpus, trainLabelsRaw, maxTrainLength = readCoNLL(train_path, charVocabulary, targetVocabulary, args.ner_tag_field, embedding_vocab)
		print("Train Corpus contains " + str(len(trainCorpus)) + " sentences and maximum sentence length is " + str(maxTrainLength))

		trainCorpusRawSorted = trainCorpus
		trainLabelsRawSorted = trainLabelsRaw
		print("Sorted the train Corpus based on length ")

		devCorpus, devLabelsRaw, maxDevLength = readCoNLL(dev_path, charVocabulary, targetVocabulary, args.ner_tag_field, embedding_vocab)
		print("Dev Corpus contains " + str(len(devCorpus)) + " sentences and maximum sentence length is " + str(maxDevLength))

		testCorpus, testLabelsRaw, maxTestLength = readCoNLL(test_path, charVocabulary, targetVocabulary, args.ner_tag_field, embedding_vocab)
		print("Test Corpus contains " + str(len(testCorpus)) + " sentences and maximum sentence length is " + str(maxTestLength))

	if not targetVocabulary.get_freeze():
		print(targetVocabulary._tok_to_ind)

		tmp_filename = '%s/output.vocab' % (save_dir)
		with open(tmp_filename, "w") as f:
			json.dump(targetVocabulary.__dict__, f)
		targetVocabulary.set_freeze()

	if not charVocabulary.get_freeze():
		tmp_filename = '%s/char.vocab' % (save_dir)
		with open(tmp_filename, "w") as f:
			json.dump(charVocabulary.__dict__, f)
		charVocabulary.set_freeze()

	embeddingDimension = args.embedDimension
	word_embedding = np.random.uniform(-0.1, 0.1, (inputVocabulary.__len__(), embeddingDimension) )
	if args.embedding_vectors:
		for everyWord in inputVocabulary._tok_to_ind:
			if everyWord in embedding_vocab:
				word_embedding[ inputVocabulary.__get_word__(everyWord) ] = embedd_dict[embedding_vocab[everyWord]]

		tmp_filename = '%s/input.vocab' % (save_dir)
		with open(tmp_filename, "w") as f:
			json.dump(inputVocabulary.__dict__, f)
		inputVocabulary.set_freeze()

		del embedd_dict
		del reverse_word_vocab
		del vocabularySize
		del embedding_vocab

	print("Read " + str(targetVocabulary.__len__()) + " number of target words")
	print("Read " + str(inputVocabulary.__len__()) + " number of input words")
	print("Read " + str(charVocabulary.__len__()) + " number of characters")

	print("Number of epochs = " +  str(num_epochs))
	print("Mini-Batch size = " +  str(batch_size))
	print("Bi-LSTM Hidden size = " +  str(hidden_size))
	print("Features per CNN filter = " +  str(num_filters))
	print("Minimum ngrams for CNN filter = " +  str(min_filter_width))
	print("Maximum ngrams for CNN filter = " +  str(max_filter_width))
	print("Initial Learning Rate = " +  str(learning_rate))

	use_gpu = args.use_gpu

	network = BiCNNLSTMTranstion(inputVocabulary.__len__(), embeddingDimension, min_filter_width, max_filter_width, charVocabulary.__len__(), num_filters, hidden_size, targetVocabulary.__len__() , word_embedding, args.fineTune)

	# print(network)

	lr = learning_rate

	optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

	if not evaluation:
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

	if evaluation:
		network.load_state_dict(torch.load(save_dir + "/model"))

		tmp_filename = '%s/output.vocab.plain' % (save_dir)
		with open(tmp_filename, "w") as f:
			for index in range(len(targetVocabulary._ind_to_tok)):
				f.write(targetVocabulary._ind_to_tok[index])
				f.write("\n")
			f.close()

		tmp_filename = '%s/input.vocab.plain' % (save_dir)
		with open(tmp_filename, "w") as f:
			for index in range(len(inputVocabulary._ind_to_tok)):
				f.write(inputVocabulary._ind_to_tok[index])
				f.write("\n")
			f.close()

		tmp_filename = '%s/char.vocab.plain' % (save_dir)
		with open(tmp_filename, "w") as f:
			for index in range(len(charVocabulary._ind_to_tok)):
				f.write(charVocabulary._ind_to_tok[index])
				f.write("\n")
			f.close()

		print("Performing Evaluation")
		if args.use_gpu == 0:
			network.cpu()

		network.eval()
		tmp_filename = '%s/_test_new' % (save_dir)

		if args.use_gpu == 1:
			print("Using GPU....")
			network.cuda()

		with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
			for inputs, labels in batch(testCorpus, testLabelsRaw, 1):
				x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, inputVocabulary, targetVocabulary, charVocabulary, max_filter_width, args.use_gpu)

				loss, preds, probs = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

				count = 0

				for i in range(len(inputs)):
					for j in range(len(inputs[i])):
						writer.write(inputs[i][j] )

						for k in range(probs[i][j].size()[0]):
							writer.write(" " + str(probs[i][j][k].item()))
						writer.write(" " + inputs[i][j] + " " + labels[i][j] + " " + targetVocabulary.__get_index__( preds[i][j].item() ).upper())
						writer.write("\n")
					writer.write("\n")

			writer.close()

		acc, precision, recall, f1 = evaluate(tmp_filename, save_dir)
		print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))
	else:
		if args.use_gpu == 1:
			print("Using GPU....")
			network.cuda()

		if args.train_from:
			print("Loading pre-trained model from " + args.train_from)

			network.load_state_dict(torch.load(args.train_from))

		print("Training....")
		prev_error = 1000.0

		network.eval()
		tmp_filename = '%s/_dev' % (save_dir)
		current_epoch_loss = 0.0

		with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
			for inputs, labels in batch(devCorpus, devLabelsRaw, 1):
				x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, inputVocabulary, targetVocabulary, charVocabulary, max_filter_width, args.use_gpu)

				loss, _ = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

				current_epoch_loss = current_epoch_loss + loss.item()

				loss, preds, probs = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

				count = 0

				for i in range(len(inputs)):
					for j in range(len(inputs[i])):
						writer.write(inputs[i][j] + " " + labels[i][j] + " " + targetVocabulary.__get_index__( preds[i][j].item() ).upper())
						writer.write("\n")
					writer.write("\n")

			writer.close()

		acc, precision, recall, f1 = evaluate(tmp_filename, save_dir)
		print('dev loss: %.2f, dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (current_epoch_loss, acc, precision, recall, f1))

		for epoch in range(1, num_epochs + 1):
			print('Epoch %d ( learning rate=%.4f ): ' % (epoch, lr))

			train_err = 0.
			train_corr = 0.
			train_total = 0.

			start_time = time.time()
			num_back = 0
			network.train()

			count = 0
			count_batch = 0

			with tqdm(total= ( len(trainCorpusRawSorted)) ) as pbar:
				for inputs, labels in batch(trainCorpusRawSorted, trainLabelsRawSorted, batch_size):

					x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, inputVocabulary, targetVocabulary, charVocabulary, max_filter_width, args.use_gpu)

					optim.zero_grad()

					loss, _ = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

					loss.backward()
					optim.step()

					train_err += loss.item()
					train_total += batch_length.data.sum()

					count = count + current_batch_size
					count_batch = count_batch + 1

					time_ave = (time.time() - start_time) / count
					time_left = (num_batches - count_batch) * time_ave
					pbar.update(1)

			print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / count, time.time() - start_time))

			network.eval()
			tmp_filename = '%s/_dev%d' % (save_dir, epoch)
			current_epoch_loss = 0.0

			with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
				for inputs, labels in batch(devCorpus, devLabelsRaw, 1):
					x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, inputVocabulary, targetVocabulary, charVocabulary, max_filter_width, args.use_gpu)

					loss, _ = network.loss(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)
					current_epoch_loss = current_epoch_loss + loss.item()

					loss, preds, probs = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

					count = 0

					for i in range(len(inputs)):
						for j in range(len(inputs[i])):
							writer.write(inputs[i][j] + " " + labels[i][j] + " " + targetVocabulary.__get_index__( preds[i][j].item() ).upper())
							writer.write("\n")
						writer.write("\n")

				writer.close()

			acc, precision, recall, f1 = evaluate(tmp_filename, save_dir)
			print('dev loss: %.2f, dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (current_epoch_loss, acc, precision, recall, f1))

			if epoch > 1:
				if current_epoch_loss > prev_error:
					lr = lr * 0.7
					optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

					network.load_state_dict(torch.load(save_dir + "/model"))
					network.eval()

					if lr < 0.002:
						network.eval()
						tmp_filename = '%s/_test%d' % (save_dir, epoch)

						with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
							for inputs, labels in batch(testCorpus, testLabelsRaw, 1):
								x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, inputVocabulary, targetVocabulary, charVocabulary, max_filter_width, args.use_gpu)

								loss, preds, probs = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

								count = 0

								for i in range(len(inputs)):
									for j in range(len(inputs[i])):
										writer.write(inputs[i][j] )

										for k in range(probs[i][j].size()[0]):
											writer.write(" " + str(probs[i][j][k].item()))
										writer.write(" " + inputs[i][j] + " " + labels[i][j] + " " + targetVocabulary.__get_index__( preds[i][j].item() ).upper())
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
					tmp_filename = '%s/_test%d' % (save_dir, epoch)

					with codecs.open(tmp_filename, "w", encoding="utf8", errors="ignore") as writer:
						for inputs, labels in batch(testCorpus, testLabelsRaw, 1):
							x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev = constructBatch(inputs, labels, inputVocabulary, targetVocabulary, charVocabulary, max_filter_width, args.use_gpu)

							loss, preds, probs = network.forward(x_input, batch_length, current_batch_size, current_max_sequence_length, y_output, mask, y_prev, args.use_gpu)

							count = 0

							for i in range(len(inputs)):
								for j in range(len(inputs[i])):
									writer.write(inputs[i][j] + " " + labels[i][j] + " " + targetVocabulary.__get_index__( preds[i][j].item() ).upper())
									writer.write("\n")
								writer.write("\n")

						writer.close()

					acc, precision, recall, f1 = evaluate(tmp_filename, save_dir)
					print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))
			else:
				prev_error = current_epoch_loss
				torch.save(network.state_dict(), save_dir + "/model")



if __name__ == '__main__':
	main()
