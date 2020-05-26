import numpy as np
from tqdm import tqdm
import torch
import pdb
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.tokenizers import Token
from overrides import overrides
import random
import transformers
from typing import Iterator
from commons import TRAIN_WORLDS
from commons import MENTION_START_TOKEN, MENTION_END_TOKEN, TITLE_AND_DESC_BONDTOKEN, CLS_TOKEN, SEP_TOKEN
NEVER_SPLIT_TOKEN = [MENTION_START_TOKEN, MENTION_END_TOKEN]
from utils import simplejopen, j_str2intidx_opener, j_intidx2str_opener

class WorldsReader(DatasetReader):
    def __init__(self, args, world_name, token_indexers, tokenizer):
        super().__init__(lazy=args.allen_lazyload)
        self.args = args
        self.world_name = world_name
        print('World {0} is now being loaded...'.format(world_name))
        self.dui2idx, self.idx2dui, self.dui2title, self.dui2desc = self.from_world_name_requireddatasetloader()
        self.berttokenizer = tokenizer # self.berttokenizer_returner()
        self.token_indexers = token_indexers # self.token_indexer_returner()

    @overrides
    def _read(self, train_dev_testflag) -> Iterator[Instance]:
        mention_ids = list()
        mentions = self.from_worldname_2_mentions()
        for mention_uniq_id, mention_data in tqdm(mentions.items()):
            if self.args.debug and (int(mention_uniq_id) == self.args.debugsamplenum):
                break
            data = self.lineparser_for_mention(line=mention_data, mention_uniq_id=mention_uniq_id)
            yield self.text_to_instance(data=data)

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        context_field = TextField(data['context'], self.token_indexers)
        fields = {"context": context_field}
        fields['gold_title_and_desc_concatenated'] = TextField(data['gold_title_and_desc_concatenated'], self.token_indexers)
        fields['gold_duidx'] = ArrayField(np.array(data['gold_duidx']))
        fields['mention_uniq_id'] = ArrayField(np.array(data['mention_uniq_id']))

        return Instance(fields)

    def lineparser_for_mention(self, line, mention_uniq_id):
        raw_mention = line["raw_mention"]
        gold_dui = line["gold_dui"]
        anchored_context = line["anchored_context"]

        data = {}
        data["mention_uniq_id"] = mention_uniq_id
        data["gold_dui"] = gold_dui
        data["gold_duidx"] = self.dui2idx[gold_dui]

        # TODO: Add limit to the tokenized max context length
        anchored_context_split = [Token(split_token) for split_token in self.tokenizer_custom_noSEPandCLS(txt=' '.join(anchored_context))]

        data['context'] = anchored_context_split
        data['gold_title_and_desc_concatenated'] = self.gold_title_and_desc_concatenated_returner(gold_dui=gold_dui)

        return data

    def huggingfacename_returner(self):
        'Return huggingface modelname and do_lower_case parameter'
        if self.args.bert_name == 'bert-base-uncased':
            return 'bert-base-uncased', True
        else:
            print('Currently', self.args.bert_name, 'are not supported.')
            exit()

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

    def tokenizer_custom_noSEPandCLS(self, txt):
        original_tokens = txt.split(' ')
        new_tokens = list()

        for token in original_tokens:
            if token in NEVER_SPLIT_TOKEN:
                new_tokens.append(token)
                continue
            else:
                split_to_subwords = self.berttokenizer.tokenize(token) # token is oneword, split_tokens
                if ['[CLS]'] in split_to_subwords:
                    split_to_subwords.remove('[CLS]')
                if ['[SEP]'] in split_to_subwords:
                    split_to_subwords.remove('[SEP]')
                if split_to_subwords == []:
                    new_tokens.append('[UNK]')
                else:
                    new_tokens += split_to_subwords

        return new_tokens

    def from_world_name_requireddatasetloader(self):
        '''
        from world_name, return
        dui2desc.json  dui2desc_raw.json  dui2idx.json  dui2title.json  idx2dui.json
        :return:
        '''
        dir_for_specified_world = self.args.dir_for_each_world + self.world_name + '/'
        dui2desc_path = dir_for_specified_world + 'dui2desc.json'
        dui2idx_path = dir_for_specified_world + 'dui2idx.json'
        idx2dui_path = dir_for_specified_world + 'idx2dui.json'
        dui2title_path = dir_for_specified_world + 'dui2title.json'

        return j_str2intidx_opener(dui2idx_path), j_intidx2str_opener(idx2dui_path), simplejopen(dui2title_path), simplejopen(dui2desc_path)

    def gold_title_and_desc_concatenated_returner(self, gold_dui):
        title = self.tokenizer_custom_noSEPandCLS(txt=self.dui2title[gold_dui])
        desc = self.tokenizer_custom_noSEPandCLS(txt=self.dui2desc[gold_dui])

        concatenated = list()
        concatenated.append(CLS_TOKEN)
        concatenated += title[:self.args.max_title_len]
        concatenated.append(TITLE_AND_DESC_BONDTOKEN)
        concatenated += desc[:self.args.max_desc_len]
        concatenated.append(SEP_TOKEN)

        return [Token(tokenized_word) for tokenized_word in concatenated]

    def from_worldname_2_mentions(self):
        mentions_path = self.args.mentions_splitbyworld_dir + self.world_name + '/mentions.json'
        return simplejopen(mentions_path)

'''
For iterating all entities
'''
class OneWorldAllEntityinKBIterateLoader(DatasetReader):
    def __init__(self, args, idx2dui, dui2title, dui2desc,
                 textfield_embedder, pretrained_tokenizer, token_indexer):
        super(OneWorldAllEntityinKBIterateLoader, self).__init__(lazy=args.allen_lazyload)
        self.args = args
        self.idx2dui = idx2dui
        self.dui2title = dui2title
        self.dui2desc = dui2desc
        self.textfield_embedder = textfield_embedder
        self.pretrained_tokenizer = pretrained_tokenizer
        self.token_indexers = token_indexer

    @overrides
    def _read(self,file_path=None) -> Iterator[Instance]:
        for idx, dui in tqdm(self.idx2dui.items()):

            data = self.dui2data(dui=dui, idx=idx)
            yield self.text_to_instance(data=data)

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        title_and_desc_concatenated = TextField(data['title_and_desc_concatnated_text'], self.token_indexers)
        fields = {"title_and_desc_concatnated_text": title_and_desc_concatenated, 'duidx':ArrayField(np.array(data['duidx'], dtype='int32'))}

        return Instance(fields)

    def tokenizer_custom(self, txt):
        original_tokens = txt.split(' ')
        new_tokens = list()

        for token in original_tokens:
            split_to_subwords = self.pretrained_tokenizer.tokenize(token) # token is oneword, split_tokens
            if ['[CLS]'] in split_to_subwords:
                split_to_subwords.remove('[CLS]')
            if ['[SEP]'] in split_to_subwords:
                split_to_subwords.remove('[SEP]')
            if split_to_subwords == []:
                new_tokens.append('[UNK]')
            else:
                new_tokens += split_to_subwords

        return new_tokens

    def dui2data(self, dui, idx):
        title_and_desc_concatenated = []
        title_and_desc_concatenated.append('[CLS]')
        title = self.dui2title[dui]
        title_tokens = [Token(split_word) for split_word in self.tokenizer_custom(txt=title)]

        title_and_desc_concatenated += title_tokens[:self.args.max_title_len]

        title_and_desc_concatenated.append(TITLE_AND_DESC_BONDTOKEN)
        desc = self.dui2desc[dui]
        desc_tokens = [Token(split_word) for split_word in self.tokenizer_custom(txt=desc)]
        title_and_desc_concatenated += desc_tokens[:self.args.max_desc_len]
        title_and_desc_concatenated.append('[SEP]')

        return {'title_and_desc_concatnated_text':[Token(split_word_) for split_word_ in title_and_desc_concatenated], 'duidx': idx}
