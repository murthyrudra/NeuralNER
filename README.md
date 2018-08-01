# NeuralNER
Implementation of Multilingual Neural NER mentioned in the paper "Judicious Selection of Training Data in Assisting Language for Multilingual Neural NER, Rudra Murthy and Anoop Kunchukuttan and Dr. Pushpak Bhattacharyya, ACL 2018 , Melbourne, July 15-20, 2018"

Training Steps:

python trainNERTransition.py --embedding_dict=<path to word embedding file> --train=<path to train file in CoNLL format> --dev=<path to development file in CoNLL format> --test=<path to test file in CoNLL format> --num_epochs=<maximum number of epochs> --learning_rate=<initial learning rate> --batch_size=<mini-batch size> --hidden_size=<bi-lstm hidden layer size> --num_filters=<number of character features extracted per filter> --min_filter_width <minimum number of character ngrams to look at> --max_filter_width <maximum number of character ngrams to look at> --use_gpu=1 --ner_tag_field=<ner tag column number> --save-dir=<save the model to this directory>
