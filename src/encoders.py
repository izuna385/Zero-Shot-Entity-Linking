'''
Seq2VecEncoders for encoding mentions and entities.
'''
import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.nn.util import get_text_field_mask, add_positional_features
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from commons import TITLE_AND_DESC_BONDTOKEN, CLS_TOKEN, SEP_TOKEN
import numpy as np
from tqdm import tqdm
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
import pdb
from allennlp.models import Model

class Pooler_for_title_and_desc(Seq2VecEncoder):
    def __init__(self, args, word_embedder):
        super(Pooler_for_title_and_desc, self).__init__()
        self.args = args
        self.huggingface_nameloader()
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

    def huggingface_nameloader(self):
        if self.args.bert_name == 'bert-base-uncased':
            self.bert_weight_filepath = 'bert-base-uncased'
        else:
            self.bert_weight_filepath = 'dummy'
            print('Currently not supported', self.args.bert_name)
            exit()

    def forward(self, title_and_desc_concatnated_text):
        mask_sent = get_text_field_mask(title_and_desc_concatnated_text)
        entity_emb = self.word_embedder(title_and_desc_concatnated_text)
        entity_emb = self.word_embedding_dropout(entity_emb)
        entity_emb = self.bertpooler_sec2vec(entity_emb, mask_sent)

        return entity_emb

class Pooler_for_mention(Seq2VecEncoder):
    def __init__(self, args, word_embedder):
        super(Pooler_for_mention, self).__init__()
        self.args = args
        self.huggingface_nameloader()
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(self.args.word_embedding_dropout)

        self.linear_for_mention_encoding = nn.Linear(self.bertpooler_sec2vec.get_output_dim(),self.bertpooler_sec2vec.get_output_dim())

    def huggingface_nameloader(self):
        if self.args.bert_name == 'bert-base-uncased':
            self.bert_weight_filepath = 'bert-base-uncased'
        else:
            self.bert_weight_filepath = 'dummy'
            print('Currently not supported', self.args.bert_name)
            exit()

    def forward(self, contextualized_mention):
        mask_sent = get_text_field_mask(contextualized_mention)
        mention_emb = self.word_embedder(contextualized_mention)
        mention_emb = self.word_embedding_dropout(mention_emb)
        mention_emb = self.bertpooler_sec2vec(mention_emb, mask_sent)

        if self.args.add_linear_for_mention:
            mention_emb = self.linear_for_mention_encoding(mention_emb)
        else:
            pass
        return mention_emb

class InKBAllEntitiesEncoder:
    def __init__(self, args, entity_loader_datasetreaderclass, entity_encoder_wrapping_model, vocab):
        self.args = args
        self.entity_loader_datasetreader = entity_loader_datasetreaderclass
        self.sequence_iterator_for_encoding_entities = BasicIterator(batch_size=128)
        self.vocab = vocab
        self.entity_encoder_wrapping_model = entity_encoder_wrapping_model
        self.entity_encoder_wrapping_model.eval()
        self.cuda_device = 0

    def encoding_all_entities(self):
        duidx2emb = {}
        ds = self.entity_loader_datasetreader.read('test')
        self.sequence_iterator_for_encoding_entities.index_with(self.vocab)
        entity_generator = self.sequence_iterator_for_encoding_entities(ds, num_epochs=1, shuffle=False)
        entity_generator_tqdm = tqdm(entity_generator, total=self.sequence_iterator_for_encoding_entities.get_num_batches(ds))
        print('======Encoding all entites from title and description=====')
        with torch.no_grad():
            for batch in entity_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                duidxs, embs = self._extract_cuidx_and_its_encoded_emb(batch)
                for duidx, emb in zip(duidxs, embs):
                    duidx2emb.update({int(duidx):emb})

        return duidx2emb

    def tonp(self, tsr):
        return tsr.detach().cpu().numpy()

    def _extract_cuidx_and_its_encoded_emb(self, batch) -> np.ndarray:
        out_dict = self.entity_encoder_wrapping_model(**batch)
        return self.tonp(out_dict['gold_duidx']), self.tonp(out_dict['emb_of_entities_encoded'])

class BiEncoder_for_Eval(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder

    def forward(self, context, gold_title_and_desc_concatenated, gold_duidx, mention_uniq_id):
        contextualized_mention = self.mention_encoder(context)
        output = {'mention_uniq_id': mention_uniq_id,
                  'gold_duidx': gold_duidx,
                  'contextualized_mention': contextualized_mention}

        return output
