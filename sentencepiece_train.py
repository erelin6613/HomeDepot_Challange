import os
from sentencepiece import (SentencePieceTrainer, 
	SentencePieceProcessor)

def train_sp_model(text_file):
	sp_model = SentencePieceTrainer.train(input=text_file, 
		vocab_size=32000, model_type='word', 
		hard_vocab_limit=False,
		model_prefix='m')

def tokens_to_ids(text, model_file=None):
	if model_file is None:
		if not os.path.isfile('m.model'):
			raise Exception('No SentencePiece model file is found.\
				Consider training one first.')
		else:
			sp = SentencePieceProcessor(model_file='m.model')
	else:
		sp = SentencePieceProcessor(model_file=model_file)
	enc = [x[0] for x in sp.encode(text)]
	return enc