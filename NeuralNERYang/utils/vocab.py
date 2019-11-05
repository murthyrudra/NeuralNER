from __future__ import print_function

__author__ = 'rudramurthy'


class Vocab(object):

    def __init__(self):
        super().__init__()

        self._tok_to_ind = {}
        self._ind_to_tok = {}
        self._tok_counts = {}
        self.__freeze__ = False

    def add(self, word):
        word = word.lower()
        ind = self._tok_to_ind.get(word, None)

        if ind is None:
            ind = len(self._tok_to_ind)
            self._ind_to_tok[ind] = word
            self._tok_to_ind[word] = ind
            self._tok_counts[word] = 1
        else:
            self._tok_counts[word] += 1
        return ind

    def __len__(self):
        return len(self._tok_to_ind)

    def __get_word__(self, word):
        word = word.lower()
        return self._tok_to_ind.get(word, None)

    def __get_word_train__(self, word):
        word = word.lower()
        return self._tok_to_ind.get(word, self._tok_to_ind.get("<unk>"))

    def __get_index__(self, index):
        return self._ind_to_tok.get(index)

    def __is_empty__(self):
        if len(self._tok_to_ind) == 0:
            return True
        else:
            return False

    def set_freeze(self):
        self.__freeze__ = True

    def get_freeze(self):
        return self.__freeze__

    def __iter__(self):
        return iter(self._tok_to_ind)

    def process(self):
        self._ind_to_tok_temp = {}
        for key in self._ind_to_tok:
            key_int = int(key)
            self._ind_to_tok_temp[key_int] = self._ind_to_tok[key]

        self._ind_to_tok = self._ind_to_tok_temp.copy()

    def trim(self):
        self._ind_to_tok_temp = {}
        self._tok_to_ind_temp = {}

        for key in self._tok_counts:
            if self._tok_counts[key] > 10:
                ind = len(self._tok_to_ind_temp)
                self._ind_to_tok_temp[ind] = key
                self._tok_to_ind_temp[key] = ind

        self._ind_to_tok = self._ind_to_tok_temp.copy()
        self._tok_to_ind = self._tok_to_ind_temp.copy()


class CharVocab(object):

    def __init__(self):
        super().__init__()

        self._tok_to_ind = {}
        self._ind_to_tok = {}
        self._tok_counts = {}
        self.__freeze__ = False

    def add(self, word):
        ind = self._tok_to_ind.get(word, None)

        if ind is None:
            ind = len(self._tok_to_ind)
            self._ind_to_tok[ind] = word
            self._tok_to_ind[word] = ind
            self._tok_counts[word] = 1
        else:
            self._tok_counts[word] += 1
        return ind

    def __len__(self):
        return len(self._tok_to_ind)

    def __get_word__(self, word):
        return self._tok_to_ind.get(word, None)

    def __get_word_train__(self, word):
        return self._tok_to_ind.get(word, self._tok_to_ind.get("<unk>"))

    def __get_index__(self, index):
        return self._ind_to_tok.get(index)

    def __is_empty__(self):
        if len(self._tok_to_ind) == 0:
            return True
        else:
            return False

    def set_freeze(self):
        self.__freeze__ = True

    def get_freeze(self):
        return self.__freeze__

    def __iter__(self):
        return iter(self._tok_to_ind)

    def process(self):
        self._ind_to_tok_temp = {}
        for key in self._ind_to_tok:
            key_int = int(key)
            self._ind_to_tok_temp[key_int] = self._ind_to_tok[key]

        self._ind_to_tok = self._ind_to_tok_temp.copy()

    def trim(self):
        self._ind_to_tok_temp = {}
        self._tok_to_ind_temp = {}

        for key in self._tok_counts:
            if self._tok_counts[key] > 10:
                ind = len(self._tok_to_ind_temp)
                self._ind_to_tok_temp[ind] = key
                self._tok_to_ind_temp[key] = ind

        self._ind_to_tok = self._ind_to_tok_temp.copy()
        self._tok_to_ind = self._tok_to_ind_temp.copy()
