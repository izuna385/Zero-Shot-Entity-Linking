from commons import MENTION_START_TOKEN, MENTION_END_TOKEN, TITLE_AND_DESC_BONDTOKEN, CLS_TOKEN, SEP_TOKEN
NEVER_SPLIT_TOKEN = [MENTION_START_TOKEN, MENTION_END_TOKEN]
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.tokenizers import Token
import transformers

class TokenIndexerReturner:
    def __init__(self, args):
        self.args = args

    def token_indexer_returner(self):
        huggingface_name, do_lower_case = self.huggingfacename_returner()
        return {'tokens': PretrainedTransformerIndexer(
            model_name=huggingface_name,
            do_lowercase=do_lower_case)
        }

    def berttokenizer_returner(self):
        if self.args.bert_name == 'bert-base-uncased':
            vocab_file = './src/vocab_file/bert-base-uncased-vocab.txt'
            do_lower_case = True
        else:
            print('currently not supported:', self.args.bert_name)
            raise NotImplementedError
        return transformers.BertTokenizer(vocab_file=vocab_file,
                                          do_lower_case=do_lower_case,
                                          do_basic_tokenize=True,
                                          never_split=NEVER_SPLIT_TOKEN)

    def huggingfacename_returner(self):
        'Return huggingface modelname and do_lower_case parameter'
        if self.args.bert_name == 'bert-base-uncased':
            return 'bert-base-uncased', True
        else:
            print('Currently', self.args.bert_name, 'are not supported.')
            exit()