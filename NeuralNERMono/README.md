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

## Data
Due to licensing issue we are unable to share the data for English and Indian languages (except Marathi). For Marathi, please fill the form here http://www.cfilt.iitb.ac.in/ner/download_data.html

However, Spanish and Dutch data are freely available here https://www.clips.uantwerpen.be/conll2002/ner/. German NEr data is taken from https://nlpado.de/~sebastian/software/ner_german.shtml. Also, we make a copy of Spanish, Dutch and German languages data available in the Data folder.


## Results
|  Language |                        Dataset Link                       |                             Word Embeddings                             |                      Reference                     | F1-Score |
|:---------:|:---------------------------------------------------------:|:-----------------------------------------------------------------------:|:--------------------------------------------------:|:--------:|
| English   | CoNLL 2003 https://www.clips.uantwerpen.be/conll2003/ner/ | Spectral Embeddings  http://www.pdhillon.com/code.html                  | https://arxiv.org/abs/1607.00198                   |    90.94 |
| Spanish   | CoNLL 2002 https://www.clips.uantwerpen.be/conll2002/ner/ | Spectral Embeddings                                                     | https://arxiv.org/abs/1607.00198                   |    85.75 |
| Dutch     | CoNLL 2002 https://www.clips.uantwerpen.be/conll2002/ner/ | Spectral Embeddings                                                     | https://arxiv.org/abs/1607.00198                   |    85.20 |
| German    | https://nlpado.de/~sebastian/software/ner_german.shtml    | Spectral Embeddings                                                     | https://aclanthology.info/papers/P18-2064/p18-2064 |    87.64 |
| Italian   | EVALITA 2009                                              | Spectral Embeddings                                                     | https://aclanthology.info/papers/P18-2064/p18-2064 |    75.98 |
| Hindi     | FIRE 2014                                                 | Fasttext Embeddings https://fasttext.cc/docs/en/pretrained-vectors.html | https://aclanthology.info/papers/P18-2064/p18-2064 |    64.93 |
| Marathi   | FIRE 2014                                                 | Fasttext Embeddings                                                     | https://aclanthology.info/papers/P18-2064/p18-2064 |    61.46 |
| Bengali   | FIRE 2014                                                 | Fasttext Embeddings                                                     | https://aclanthology.info/papers/P18-2064/p18-2064 |    55.61 |
| Malayalam | FIRE 2014                                                 | Fasttext Embeddings                                                     | https://aclanthology.info/papers/P18-2064/p18-2064 |    64.59 |
| Tamil     | FIRE 2014                                                 | Fasttext Embeddings                                                     | https://aclanthology.info/papers/P18-2064/p18-2064 |    65.39 |

## PPS: The reason for difference in monolingual NER performance for Bengali, Tamil and Malayalam are due to certain pre-processing steps which were not performed in the ACL 2018 paper. We have observed that some of the sentences have length greater than 200 words. Manually splitting these longer sentences into smaller ones using '.' as delimiter lead to substantial imrpovement
