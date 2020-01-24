from __future__ import print_function

__author__ = 'rudramurthy'


class Vocab(object):
	"""
	A class used to represent dictionary of strings

	...

	Attributes
	----------
	_tok_to_ind : dict
		a dictionary of string object to index mapping
	_ind_to_tok : dict
		a dictionary of index to string object mapping
	_tok_counts : dict
		a dictionary of string object and its corresponding frequency in the document
	_freeze : bool
		no more string objects can be added
	

	Methods
	-------
	add(word=None)
		Adds a new string object to the dictionary if not present
	len()
		Total number of string objects in the dictionary
	__get_word__(word=None)
		Given a string object, get the corresponding index
	__get_word_train__(word=None)
		Given a string object, get the corresponding index. If not present return index of <unk> object
	__get_index__(index=None)
		Given an index, get the word at that index
	__is_empty__()
		Return is the dictionary empty is empty or not
	get_freeze()
		Returns the value of _freeze parameter
	set_freeze()
		Set the _freeze parameter to True
	__iter__()
		Returns the iterator to the dictionary
	process()
		Re-assign the index to word mappings. Is useful when any string object is deleted from the dictionary
	trim()
		Remove word appearing less than the specified threshold
	"""

	def __init__(self):
		"""" 

		"""
		super().__init__()

		self._tok_to_ind = {}
		self._ind_to_tok = {}
		self._tok_counts = {}
		self.__freeze__ = False

	def add(self, word):
		"""" Add a word to the dictionary if not present

		Parameters
		----------
		word :  
			The word to be added

		"""

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
		"""" Return the number of objects in the dictionary

		"""
		return len(self._tok_to_ind)

	def __get_word__(self, word):
		"""" Get the index of the given word

		Parameters
		----------
		word :  
			The word whose index is required. If present return index else None

		"""

		word = word.lower()
		return self._tok_to_ind.get(word, None)

	def __get_word_train__(self, word):
		"""" Get the index of the given word

		Parameters
		----------
		word :  
			The word whose index is required. If present return index else return index of <unk> string

		"""

		word = word.lower()
		return self._tok_to_ind.get(word, self._tok_to_ind.get("<unk>"))

	def __get_index__(self, index):
		"""" Get the word at the specified index

		Parameters
		----------
		index :  
			The index 

		"""

		return self._ind_to_tok.get(index)

	def __is_empty__(self):
		"""" Is the dictionary empty
		
		"""

		if len(self._tok_to_ind) == 0:
			return True
		else:
			return False

	def set_freeze(self):
		"""" Set the __freeze__ variable to True

		"""
		self.__freeze__ = True

	def get_freeze(self):
		"""" Get the value of the __freeze__ variable

		"""

		return self.__freeze__

	def __iter__(self):
		"""" Return the iterator to the dictionary

		"""

		return iter(self._tok_to_ind)

	def process(self):
		"""" Reassign word to index mappings

		"""

		self._ind_to_tok_temp = {}
		for key in self._ind_to_tok:
			key_int = int(key)
			self._ind_to_tok_temp[key_int] = self._ind_to_tok[key]

		self._ind_to_tok = self._ind_to_tok_temp.copy()

	def trim(self):
		"""" Drop words appearing less with frequency less than 10

		"""

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
	"""
	A class used to represent dictionary of characters

	...

	Attributes
	----------
	_tok_to_ind : dict
		a dictionary of string object to index mapping
	_ind_to_tok : dict
		a dictionary of index to string object mapping
	_tok_counts : dict
		a dictionary of string object and its corresponding frequency in the document
	_freeze : bool
		no more string objects can be added
	

	Methods
	-------
	add(word=None)
		Adds a new string object to the dictionary if not present
	len()
		Total number of string objects in the dictionary
	__get_word__(word=None)
		Given a string object, get the corresponding index
	__get_word_train__(word=None)
		Given a string object, get the corresponding index. If not present return index of <unk> object
	__get_index__(index=None)
		Given an index, get the word at that index
	__is_empty__()
		Return is the dictionary empty is empty or not
	get_freeze()
		Returns the value of _freeze parameter
	set_freeze()
		Set the _freeze parameter to True
	__iter__()
		Returns the iterator to the dictionary
	process()
		Re-assign the index to word mappings. Is useful when any string object is deleted from the dictionary
	trim()
		Remove word appearing less than the specified threshold
	"""

	def __init__(self):
		"""" 

		"""

		super().__init__()

		self._tok_to_ind = {}
		self._ind_to_tok = {}
		self._tok_counts = {}
		self.__freeze__ = False

	def add(self, word):
		"""" Add a word to the dictionary if not present

		Parameters
		----------
		word :  
			The word to be added

		"""

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
		"""" Return the number of objects in the dictionary

		"""

		return len(self._tok_to_ind)

	def __get_word__(self, word):
		"""" Get the index of the given word

		Parameters
		----------
		word :  
			The word whose index is required. If present return index else None

		"""

		return self._tok_to_ind.get(word, None)

	def __get_word_train__(self, word):
		"""" Get the index of the given word

		Parameters
		----------
		word :  
			The word whose index is required. If present return index else return index of <unk> string

		"""

		return self._tok_to_ind.get(word, self._tok_to_ind.get("<unk>"))

	def __get_index__(self, index):
		"""" Get the word at the specified index

		Parameters
		----------
		index :  
			The index 

		"""

		return self._ind_to_tok.get(index)
	
	def __is_empty__(self):
		"""" Is the dictionary empty
		
		"""

		if len(self._tok_to_ind) == 0:
			return True
		else:
			return False

	def set_freeze(self):
		"""" Set the __freeze__ variable to True

		"""
		self.__freeze__ = True

	def get_freeze(self):
		"""" Get the value of the __freeze__ variable

		"""

		return self.__freeze__

	def __iter__(self):
		"""" Return the iterator to the dictionary

		"""

		return iter(self._tok_to_ind)

	def process(self):
		"""" Reassign word to index mappings

		"""

		self._ind_to_tok_temp = {}
		for key in self._ind_to_tok:
			key_int = int(key)
			self._ind_to_tok_temp[key_int] = self._ind_to_tok[key]

		self._ind_to_tok = self._ind_to_tok_temp.copy()

	def trim(self):
		"""" Drop words appearing less with frequency less than 10

		"""

		self._ind_to_tok_temp = {}
		self._tok_to_ind_temp = {}

		for key in self._tok_counts:
			if self._tok_counts[key] > 10:
				ind = len(self._tok_to_ind_temp)
				self._ind_to_tok_temp[ind] = key
				self._tok_to_ind_temp[key] = ind

		self._ind_to_tok = self._ind_to_tok_temp.copy()
		self._tok_to_ind = self._tok_to_ind_temp.copy()
