__author__ = 'rudramurthy'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class SubwordModule(nn.Module):

    def __init__(self, ngrams, inputDim, outDim):
        super(SubwordModule, self).__init__()

        self.ngrams = ngrams
        self.inputDim = inputDim

        self.outDim = outDim

# in_channels is the same as rgb equivalent in vision
# for nlp in_channels is 1 unless we are using multiple feature representations
# out_channels is the number of features extracted by cnn layer
# kernel_size is dimension of the cnn filter
# as we are flattening the input representation we need to specify stride as number of input feature dimension

        self.conv = nn.Conv1d(in_channels=1, out_channels=self.outDim, kernel_size= self.ngrams * self.inputDim, stride = self.inputDim)

    def forward(self, x):
# get the convolved output which is of size miniBatch , out_channels , (reduced_input_character_length)

        x_conv = self.conv(x)
# reshape the output into miniBatch , out_channels * (reduced_input_character_length)
        x_convOut = x_conv.view(x_conv.size()[0], x_conv.size()[1] * x_conv.size()[2])
# apply max_pool1d with stride as reduced_input_character_length
        x = F.max_pool1d(x_convOut.unsqueeze(1), x_conv.size()[2])
        return x

class OutputLayer(nn.Module):
    def __init__(self, inputDimension, outputDimension ):

        super(OutputLayer,self).__init__()

        self.inputDimension = inputDimension
        self.outputDimension = outputDimension

        self.linear = nn.Linear(self.inputDimension, self.outputDimension, bias = False)

    def forward(self, x_in):

        output = self.linear(x_in)
        return output

class BiCNNLSTMTranstion(nn.Module):
    def __init__(self, vocabularySize, embedDimension, minNgrams, maxNgrams, charInputDim, charOutDim, hiddenDim, tagSize, init_embedding, tagSizeAux ):
        super(BiCNNLSTMTranstion,self).__init__()

        self.vocabularySize = vocabularySize
        self.embedDimension = embedDimension

        self.minNgrams = minNgrams
        self.maxNgrams = maxNgrams
        self.charInputDim = charInputDim
        self.charOutDim = charOutDim

        self.bilstmInputDim = self.embedDimension + (self.maxNgrams - self.minNgrams + 1) * self.charOutDim
        self.hiddenDim = hiddenDim
        self.tagSize = tagSize
        self.tagSizeAux = tagSizeAux

        self.embedLayer = nn.Embedding(self.vocabularySize, self.embedDimension, padding_idx=0)
        self.embedLayer.weight = Parameter(torch.Tensor(init_embedding))

        self.embedLayerAux = nn.Embedding(self.vocabularySize, self.embedDimension, padding_idx=0)
        self.embedLayerAux.weight = Parameter(torch.Tensor(init_embedding))

        self.charLayers = nn.ModuleList( [SubwordModule(i, self.charInputDim, self.charOutDim) for i in range(self.minNgrams, self.maxNgrams + 1) ])

        self.dropout_in = nn.Dropout()

        self.bilstm = nn.LSTM(self.bilstmInputDim, self.hiddenDim, 1, batch_first = True, bidirectional = True)
        self.bilstmAux = nn.LSTM(self.bilstmInputDim, self.hiddenDim, 1, batch_first = True, bidirectional = True)

        self.outputLayer = OutputLayer(self.hiddenDim * 2 + tagSize, tagSize)
        self.outputLayerAux = OutputLayer(self.hiddenDim * 2 + tagSizeAux, tagSizeAux)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.logsoftmaxAux = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(size_average=True, reduce=False)

    def loss(self, x, length_of_sequence, batchSize, maxLength, target, mask, y_prev, languageId, use_gpu):

        if languageId == 0:
            embedOut = self.embedLayer(x[0])
            charOut = []
    # extract the sub-word features and the input is batchSize, sequenceLength, (numberOfCharacters * characterFeatures)
            for i,l in enumerate(self.charLayers):
                charOut.append(l(x[1]))

            # concatenate all extracted character features based on the last dimension
            finalCharOut = torch.cat( [charOut[i] for i,l in enumerate(self.charLayers)], 2)

            finalCharOut = finalCharOut.view(batchSize, maxLength, finalCharOut.size()[2])

    # concatenate word representation and subword features
            finalWordOut = torch.cat( [finalCharOut, embedOut], 2)

            finalWordOut = self.dropout_in(finalWordOut)

    # convert the list of lengths into a Tensor
            seq_lengths = length_of_sequence
    # get the sorted list of lengths and the correpsonding indices
            sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

            _, rev_order = torch.sort(sorted_index)
    # convert the given input into sorted order based on sorted indices
            if use_gpu == 1:
                rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
                correctLabels = target.index_select(0, sorted_index.cuda())
            else:
                rnn_input = finalWordOut.index_select(0, sorted_index)
                correctLabels = target.index_select(0, sorted_index)

    # pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
            x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

    # pack the correct target variables so that the loss function can be appropriately calculated
            t_out = torch.nn.utils.rnn.pack_padded_sequence(correctLabels, sorted_length.data.tolist(), batch_first=True)

    # pack the correct previous target variables so that the loss function can be appropriately calculated
            t_out_prev = torch.nn.utils.rnn.pack_padded_sequence(y_prev, sorted_length.data.tolist(), batch_first=True)

            seq_output, hn = self.bilstm(x)

            inputToDecoder = torch.cat([seq_output.data, t_out_prev.data], 1)

            outputScores = self.outputLayer(inputToDecoder)
            prob_output = self.logsoftmax(outputScores)

            prob_out_correct_order_packed = torch.nn.utils.rnn.PackedSequence(prob_output, x.batch_sizes)

            prob_out_correct_order, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(prob_out_correct_order_packed, batch_first=True)

            pred, predIndex = torch.max(prob_out_correct_order, dim=2 )

            count = 0
            for i in range(batchSize):
                for j in range(length_of_sequence[i].item()):
                    count = count + 1

            return self.nll_loss(prob_output, t_out.data).sum() / count, predIndex.data
        else:
            embedOut = self.embedLayerAux(x[0])
            charOut = []
    # extract the sub-word features and the input is batchSize, sequenceLength, (numberOfCharacters * characterFeatures)
            for i,l in enumerate(self.charLayers):
                charOut.append(l(x[1]))

            # concatenate all extracted character features based on the last dimension
            finalCharOut = torch.cat( [charOut[i] for i,l in enumerate(self.charLayers)], 2)

            finalCharOut = finalCharOut.view(batchSize, maxLength, finalCharOut.size()[2])

    # concatenate word representation and subword features
            finalWordOut = torch.cat( [finalCharOut, embedOut], 2)

            finalWordOut = self.dropout_in(finalWordOut)

    # convert the list of lengths into a Tensor
            seq_lengths = length_of_sequence
    # get the sorted list of lengths and the correpsonding indices
            sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

            _, rev_order = torch.sort(sorted_index)
    # convert the given input into sorted order based on sorted indices

            if use_gpu == 1:
                rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
                correctLabels = target.index_select(0, sorted_index.cuda())
            else:
                rnn_input = finalWordOut.index_select(0, sorted_index)
                correctLabels = target.index_select(0, sorted_index)

    # pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
            x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

    # pack the correct target variables so that the loss function can be appropriately calculated
            t_out = torch.nn.utils.rnn.pack_padded_sequence(correctLabels, sorted_length.data.tolist(), batch_first=True)

    # pack the correct previous target variables so that the loss function can be appropriately calculated
            t_out_prev = torch.nn.utils.rnn.pack_padded_sequence(y_prev, sorted_length.data.tolist(), batch_first=True)

            seq_output, hn = self.bilstmAux(x)

            inputToDecoder = torch.cat([seq_output.data, t_out_prev.data], 1)

            outputScores = self.outputLayerAux(inputToDecoder)
            prob_output = self.logsoftmaxAux(outputScores)

            prob_out_correct_order_packed = torch.nn.utils.rnn.PackedSequence(prob_output, x.batch_sizes)

            prob_out_correct_order, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(prob_out_correct_order_packed, batch_first=True)

            pred, predIndex = torch.max(prob_out_correct_order, dim=2 )

            count = 0
            for i in range(batchSize):
                for j in range(length_of_sequence[i].item()):
                    count = count + 1

            return self.nll_loss(prob_output, t_out.data).sum() / count * 0.05, predIndex.data

    def forward(self, x, length_of_sequence, batchSize, maxLength, target, mask, y_prev, languageId, use_gpu):
# batchSize is always 1, it's an hack but still we'll go with it
        embedOut = []
        embedOut = self.embedLayer(x[0])

        charOut = []
# extract the sub-word features and the input is batchSize, sequenceLength, (numberOfCharacters * characterFeatures)
        for i,l in enumerate(self.charLayers):
            charOut.append(l(x[1]))

        # concatenate all extracted character features based on the last dimension
        finalCharOut = torch.cat( [charOut[i] for i,l in enumerate(self.charLayers)], 2)

        finalCharOut = finalCharOut.view(batchSize, maxLength, finalCharOut.size()[2])

# concatenate word representation and subword features
        finalWordOut = torch.cat( [finalCharOut, embedOut], 2)

        finalWordOut = self.dropout_in(finalWordOut)

# convert the list of lengths into a Tensor
        seq_lengths = length_of_sequence
# get the sorted list of lengths and the correpsonding indices
        sorted_length, sorted_index = torch.sort(seq_lengths, dim=0, descending=True)

        _, rev_order = torch.sort(sorted_index)
# convert the given input into sorted order based on sorted indices
        if use_gpu == 1:
            rnn_input = finalWordOut.index_select(0, sorted_index.cuda())
            correctLabels = target.index_select(0, sorted_index.cuda())
        else:
            rnn_input = finalWordOut.index_select(0, sorted_index)
            correctLabels = target.index_select(0, sorted_index)

# pack the sequence with pads so that Bi-LSTM returns zero for the remaining entries
        x = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_length.data.tolist(), batch_first=True)

# forward pass to get the output from Bi-LSTM layer
        seq_output, hn = self.bilstm(x)

        seq_bilstm_out_order, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(seq_output, batch_first=True)

        count = 0
        for i in range(batchSize):
            for j in range(length_of_sequence[i].item()):
                count = count + 1

        # loss = self.nll_loss(prob_output, t_out.data).sum() / count

        loss, predIndex = self._decode(seq_bilstm_out_order, target, batchSize,  maxLength, mask)

        return loss, predIndex

    def _decode(self, seq_output, correctTarget, batchSize, maxLength, mask, use_gpu):
        predictionList = torch.LongTensor(batchSize, maxLength).fill_(0)

        loss = 0.0

        for time_step in range(maxLength):
            init_predictions = torch.FloatTensor(batchSize, self.tagSize).fill_(0.0)

            if time_step != 0 :
                # for everyWord in batch for that time-step
                for j in range(batchSize):
                    init_predictions[j][ predictionList[j][time_step-1] ] = 1.0

            # input to output layer is both bi-lstm features and correct previous tag
            if use_gpu == 1:
                inputToOutLayerWord = torch.cat([seq_output[:,time_step,:], Variable(init_predictions.cuda())], 1)
            else:
                inputToOutLayerWord = torch.cat([seq_output[:,time_step,:], Variable(init_predictions)], 1)

            wordPrePrediction = self.outputLayer(inputToOutLayerWord)
            wordPrediction = self.logsoftmax(wordPrePrediction)
            # batch * tagSize

            count = 0
            # for every batch
            for j in range(batchSize):
                # number of non-padded words in that time-sequence
                if mask[j][time_step].item() == 1.0:
                    count = count + 1

            # need to calculate loss only for the non-padded words
            newWordScores = torch.FloatTensor(count, self.tagSize).fill_(0.0)
            newTarget = torch.LongTensor(count).fill_(0)
            k = 0

            # for every batch
            for j in range(batchSize):
                if mask[j][time_step].item() == 1.0:
                    newWordScores[k] = wordPrediction[j].data[0]
                    newTarget[k] = correctTarget[j][time_step].item()
                    k = k + 1

            loss = loss + self.nll_loss(Variable(newWordScores), Variable(newTarget) ).sum()/count

            value, index = torch.max(wordPrediction, 1)

            for j in range(batchSize):
                predictionList[j][time_step] = index[j].item()

        return loss, predictionList
