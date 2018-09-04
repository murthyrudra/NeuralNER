# NeuralNER

This repository implements monolingual Sequence Labeler using Hierarchical Neural Networks mentioned in the paper "Judicious Selection of Training Data in Assisting Language for Multilingual Neural NER, ACL 2018". The code is based on PyTorch. The original version of the code was written in Torch (https://github.com/murthyrudra/SequenceLabelerDeepLearning). The Torch version implements the beam search described in the paper whereas the PyTorch version takes the label with the highest probability value as the predicted label.

## Training Steps:

```sh
python NeuralNER.py --embedding_dict="path to word embedding file" --train="path to train file in CoNLL format" --dev="path to development file in CoNLL format" --test="path to test file in CoNLL format" --num_epochs="maximum number of epochs" --learning_rate="initial learning rate" --batch_size="mini-batch size" --hidden_size="bi-lstm hidden layer size" --num_filters="number of character features extracted per filter" --min_filter_width "minimum number of character ngrams to look at" --max_filter_width "maximum number of character ngrams to look at" --use_gpu=1 --ner_tag_field="ner tag column number" --save-dir="save the model to this directory"
```

The model saves the character vocabulary and the label vocabulary created in save-dir directory under the name "char.vocab" and "tag.vocab".

One can also specify the character and tag vocabulary by providing the options "--vocabChar" and "--vocabTag".

## Performing evaluation

Training and development data are not required for evaluating the pre-trained model. Just specify the test data and character and tag vocabularies.

```sh
python NeuralNER.py --embedding_dict="path to word embedding file" --test="path to test file in CoNLL format" --hidden_size="bi-lstm hidden layer size" --num_filters="number of character features extracted per filter" --min_filter_width "minimum number of character ngrams to look at" --max_filter_width "maximum number of character ngrams to look at" --use_gpu=1 --ner_tag_field="ner tag column number" --save-dir="directory in which model is saved" --vocabChar="directory in which model is saved"/char.vocab --vocabTag="directory in which model is saved"/tag.vocab --perform_evaluation=True
```

## For Deploying
The pretrained model can also be used to obtain named entities on unlabeled corpus. The unlabeled corpus contains every sentence in it's own line. Please run the following command to obtaining named entities on any unlabeled corpus

```sh
python NeuralNER.py --embedding_dict="path to word embedding file" --test="path to plain corpus" --hidden_size="bi-lstm hidden layer size" --num_filters="number of character features extracted per filter" --min_filter_width "minimum number of character ngrams to look at" --max_filter_width "maximum number of character ngrams to look at" --use_gpu=1 --ner_tag_field="ner tag column number" --save-dir="directory in which model is saved" --vocabChar="directory in which model is saved"/char.vocab --vocabTag="directory in which model is saved"/tag.vocab --perform_evaluation=True --deploy True
```


