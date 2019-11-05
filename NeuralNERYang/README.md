# NeuralNER

This repository implements multilingual Sequence Labeler using Hierarchical Neural Networks mentioned in the paper "Transfer Learning for Sequence Tagging with Hierarchical Recurrent Networks. Yang, Zhilin, Ruslan Salakhutdinov, and William W. Cohen". The code is based on PyTorch. I had implemented the earlier version of the code in Torch (https://github.com/murthyrudra/SequenceLabelerDeepLearning). The Torch version implements the beam search described in the paper whereas the PyTorch version takes the label with the highest probability value as the predicted label.

## Training Steps:

```sh
python NeuralNER.py --embedding_dict="path to word embedding file" --train="path to train file in CoNLL format" --trainAux="path to auxiliary language train file in CoNLL format" --dev="path to development file in CoNLL format" --test="path to test file in CoNLL format" --num_epochs="maximum number of epochs" --learning_rate="initial learning rate" --batch_size="mini-batch size" --hidden_size="bi-lstm hidden layer size" --num_filters="number of character features extracted per filter" --min_filter_width "minimum number of character ngrams to look at" --max_filter_width "maximum number of character ngrams to look at" --use_gpu=1 --ner_tag_field_l1="ner tag column number of source language" --ner_tag_field_l2="ner tag column number of assisting language" --save_dir="save the model to this directory"
```

The model saves the character vocabulary and the label vocabulary created in save-dir directory under the name "char.vocab" and "tag.vocab".

One can also specify the character and tag vocabulary by providing the options "--vocabChar" and "--vocabTag".
