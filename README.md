# NeuralNER
Implementation of Multilingual Neural NER mentioned in the paper "Judicious Selection of Training Data in Assisting Language for Multilingual Neural NER, Rudra Murthy and Anoop Kunchukuttan and Dr. Pushpak Bhattacharyya, ACL 2018 , Melbourne, July 15-20, 2018"


Please read the Readme file here https://github.com/murthyrudra/NeuralNER/blob/master/NeuralNERMono/README.md on how to train the network. The multilingual learning code works but needs some polishing and there might be some errors. If any please raise an issue and I will immediately address the issue.

## Important:
Please use a batch-size of 1. I found a bug with the code during test time when the batch size is greater than 1. To be safer, please use a batch-size of 1.

## To-Do

- [x] NeuralNERMono code is polished. Update NeuralNERYang and NeuralNERAllShared with similar code standards
- [ ] The system gives erroneous output when batch size is greater than 1.

## Note:
Parts of the code are borrowed from NeuroNLP2 https://github.com/XuezheMax/NeuroNLP2/. Because of their code it was easier for me to convert my earlier Torch implementation to PyTorch.
