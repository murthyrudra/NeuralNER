__author__ = 'rudramurthy'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class WordRepresentation(nn.Module):
    def __init__(self, vocabularySize, embedDimension, minNgrams, maxNgrams, charInputDim, charOutDim , init_embedding):

        super(WordRepresentation,self).__init__()

        self.vocabularySize = vocabularySize
        self.embedDimension = embedDimension

        self.minNgrams = minNgrams
        self.maxNgrams = maxNgrams
        self.charInputDim = charInputDim
        self.charOutDim = charOutDim

        self.embedLayer = nn.Embedding(self.vocabularySize, self.embedDimension, padding_idx=0)
        self.embedLayer.weight = Parameter(torch.Tensor(init_embedding))

        self.charLayers = nn.ModuleList( [SubwordModule(i, self.charInputDim, self.charOutDim) for i in range(self.minNgrams, self.maxNgrams + 1) ])

        self.dropout_in = nn.Dropout()

    def forward(self,x):
        charOut = []
# extract the sub-word features and the input is batchSize, sequenceLength, (numberOfCharacters * characterFeatures)
        for i,l in enumerate(self.charLayers):
            charOut.append(l(x[1]))

# concatenate all extracted character features based on the last dimension
        finalCharOut = torch.cat( [charOut[i] for i,l in enumerate(self.charLayers)], 2)

# miniBatch * sequenceLength * embeddingDimension
        embedOut = self.embedLayer(x[0])
# concatenate word representation and subword features
        finalWordOut = torch.cat( [finalCharOut, embedOut], 2)

        finalWordOut = self.dropout_in(finalWordOut)

        return embedOut, finalCharOut, finalWordOut

class WordInstanceRepresentation(nn.Module):
    def __init__(self, inputDimension, outputDimension ):

        super(WordInstanceRepresentation,self).__init__()

        self.inputDimension = inputDimension
        self.outputDimension = outputDimension

        self.bilstm = nn.LSTM(self.inputDimension, self.outputDimension, 1, batch_first = True, bidirectional = True)

    def forward(self, x_in, length, batchSize, maxLength):
# convert input batchSize * sequenceLength * numberOfFeatures into a variable
        batch_in = x_in

# convert the list of lengths into a Tensor
        seq_lengths = length
# get the sorted list of lengths and the correpsonding indices
        sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

        _, rev_order = torch.sort(sorted_index)
# convert the given input into sorted order based on sorted indices
        rnn_input = batch_in.index_select(0, sorted_index)
        # sorted_length = sorted_length.tolist()
# pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
        x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)
# forward pass to get the output from Bi-LSTM layer
        seq_output, hn = self.bilstm(x)
# unpack to get the final output
        tokenRep, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(seq_output, batch_first=True)

        tokenRep = tokenRep.index_select(0, rev_order)

        return tokenRep, rev_order



class BiLSTM(nn.Module):
    def __init__(self, vocabularySize, embedDimension, hiddenDim, tagSize, init_embedding ):
        super(BiLSTM,self).__init__()

        self.vocabularySize = vocabularySize
        self.embedDimension = embedDimension

        self.bilstmInputDim = self.embedDimension
        self.hiddenDim = hiddenDim
        self.tagSize = tagSize

        self.embedLayer = nn.Embedding(self.vocabularySize, self.embedDimension, padding_idx=0)
        self.embedLayer.weight = Parameter(torch.Tensor(init_embedding))

        self.bilstm = nn.LSTM(self.bilstmInputDim, self.hiddenDim, 1, batch_first = True, bidirectional = True)

        self.outputLayer = OutputLayer(self.hiddenDim * 2 ,  tagSize)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(size_average=True, reduce=False)

    def loss(self, x, length_of_sequence, batchSize, maxLength, target):

        finalWordOut = self.embedLayer(x)

# convert the list of lengths into a Tensor
        seq_lengths = length_of_sequence
# get the sorted list of lengths and the correpsonding indices
        sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

        _, rev_order = torch.sort(sorted_index)
# convert the given input into sorted order based on sorted indices
        rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
        correctLabels = target.index_select(0, sorted_index.cuda())

# pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
        x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

        t_out = torch.nn.utils.rnn.pack_padded_sequence(correctLabels, sorted_length.data.tolist(), batch_first=True)

# forward pass to get the output from Bi-LSTM layer
        seq_output, hn = self.bilstm(x)

        outputScores = self.outputLayer(seq_output[0])
        prob_output = self.logsoftmax(outputScores)

        prob_out_correct_order_packed = torch.nn.utils.rnn.PackedSequence(prob_output, x.batch_sizes)

        prob_out_correct_order, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(prob_out_correct_order_packed, batch_first=True)

        pred, predIndex = torch.max(prob_out_correct_order, dim=2 )

        count = 0
        for i in range(batchSize):
            for j in range(length_of_sequence[i].data[0]):
                count = count + 1

        return self.nll_loss(prob_output, t_out.data.squeeze(1)).sum() / count, predIndex, predIndex.data
