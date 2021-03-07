'''
CREATE dui2cano, dui2def from documents and iterative mentions
dui: Document ID in each world.
'''
from commons import TRAIN_WORLDS, DEV_WORLDS, TEST_WORLDS, MENTION_START_TOKEN, MENTION_END_TOKEN
import os, pdb
from tqdm import tqdm
import json
import spacy
from parameters import Params
import os
import copy
import time, random
from multiprocessing import Pool
import multiprocessing
import stanza
import glob

ALL_WORLDS = TRAIN_WORLDS + DEV_WORLDS + TEST_WORLDS
nlp = spacy.load("en_core_web_sm")

def simplejopen(json_file_path):
    with open(json_file_path, 'r') as f:
        j = json.load(f)
    return j

def oneworld_opener(one_world_jsonpath):
    lines = []
    with open(one_world_jsonpath, 'r') as f:
        for line in f:
            json_parsed = json.loads(line)
            lines.append(json_parsed)

    return lines

def mentions_in_train_dev_test_loader(mention_jsonpath, train_dev_testflag, mentions_dir,
                                      worldName_2_dui2rawtext):
    if not os.path.exists(mentions_dir + train_dev_testflag + '/'):
        os.mkdir(mentions_dir + train_dev_testflag)

    source_dirpath = mentions_dir + train_dev_testflag + '/'

    lines = []
    with open(mention_jsonpath, 'r') as f:
        for line in tqdm(f):
            json_parsed = json.loads(line)
            # {'category': 'MULTIPLE_CATEGORIES', 'text': 'Blink', 'end_index': 437,
            #  'context_document_id': '5A5552D0B5393B77', 'label_document_id': 'D46C64DD4AFBC7B6',
            #  'mention_id': '0000235DEE8E1F60', 'corpus': 'final_fantasy', 'start_index': 437}


            new_json = {'category': json_parsed['category'],
                        'text': json_parsed['text'],
                        'end_index': json_parsed['end_index'],
                        'context_document_id': json_parsed['context_document_id'],
                        'label_document_id': json_parsed['label_document_id'],
                        'mention_id': json_parsed['mention_id'],
                        'corpus': json_parsed['corpus'],
                        'start_index': json_parsed['start_index'],
                        'raw_context': worldName_2_dui2rawtext[json_parsed['corpus']][json_parsed["context_document_id"]].strip().split(' ')
                        }
            assert 'text' in new_json

            with open(source_dirpath + json_parsed['mention_id'] + '.json', 'w') as g:
                json.dump(new_json, g,
                          ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

    return glob.glob(source_dirpath+'*.json')

def jdump(j, path):
    with open(path, 'w') as f:
        json.dump(j, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))



def mentionConverter(one_line_mention_path):
    '''
    :param one_line_mention:
    :return: labeled_dui, mention itself, context wrapped with start/end token
    '''
    # one_line_mention = {"category": "MULTIPLE_CATEGORIES", "text": "I - Spy Express", "end_index": 842, "context_document_id": "8D5C15C034E982AF", "label_document_id": "0ABE561452932284", "mention_id": "FFF2DE1A167361C1", "corpus": "lego", "start_index": 839}
    with open(one_line_mention_path, 'r') as f:
        one_line_mention = json.load(f)
    raw_mention = one_line_mention["text"]

    gold_dui = one_line_mention["label_document_id"]
    dui_where_mention_exist = one_line_mention["context_document_id"]
    gold_world = one_line_mention["corpus"]

    mention_start_tokenidx = one_line_mention["start_index"]
    mention_end_tokenidx = one_line_mention["end_index"]

    raw_context = one_line_mention['raw_context']
    doc = nlp(' '.join(raw_context))
    sents = [sentence.text for sentence in doc.sents]
    # doc = ' '.join(raw_context)
    # sents = self.seg.segment(doc)

    sent_data = {}
    first_tok_idx = 0

    mention_insert_flag = 0

    local_mention_anchored_context = []
    for sent in sents:
        sent_length = len(sent.split(' '))
        sent_finish_idx = first_tok_idx + sent_length

        sent_idx = len(sent_data)

        sent_data.update({
            sent_idx:
                {'first_tok_idx': first_tok_idx,
                 'final_tok_idx': sent_finish_idx}
        })

        raw_sent = sent.split(' ')

        if first_tok_idx <= mention_start_tokenidx and mention_end_tokenidx < first_tok_idx + sent_length:
            mention_insert_flag += 1
            raw_sent.insert(mention_start_tokenidx - first_tok_idx, MENTION_START_TOKEN)
            raw_sent.insert(mention_end_tokenidx - first_tok_idx + 2, MENTION_END_TOKEN)
            local_mention_anchored_context = copy.copy(raw_sent)
        first_tok_idx += sent_length
        sent_data[sent_idx].update({'sents_with_anchored_if_mention_included': raw_sent})

    parsed_path = one_line_mention_path + '_parsed'
    with open(parsed_path, 'w') as g:
        json.dump(
            {"raw_mention": raw_mention,
            "gold_dui": gold_dui,
            "gold_world": gold_world,
            "anchored_context": local_mention_anchored_context,
            "mention_insert_flag": mention_insert_flag},
            g, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))
    return parsed_path

class OneWorldParser:
    '''
    Preprocess one world
    '''
    def __init__(self, args):
        self.args = args

    def oneworld_json_2_docid2title(self, one_world_json):
        dui2title, idx2dui, dui2idx = {}, {}, {}

        for idx, one_doc_data in enumerate(one_world_json):
            title = one_doc_data["title"]
            dui = one_doc_data["document_id"]

            dui2idx.update({dui:idx})
            idx2dui.update({idx:dui})
            dui2title.update({dui:title})

        return dui2title, idx2dui, dui2idx

    def oneworld_json_2_dui2desc_raw(self, one_world_json):
        dui2desc_raw = {}

        for idx, one_doc_data in enumerate(one_world_json):
            text = one_doc_data["text"]
            dui = one_doc_data["document_id"]

            dui2desc_raw.update({dui:text})

        return dui2desc_raw

    def dui2tokens_tokentrimmer(self, dui2tokens, max_token):
        dui2tokens_MaxLengthLimited = {}
        for dui, tokens in dui2tokens.items():
            dui2tokens_MaxLengthLimited.update({dui: ' '.join(tokens.strip().split(' ')[:max_token])})
        return dui2tokens_MaxLengthLimited

    def from_oneworld_dump_preprocessed_world(self, world_name):
        '''
        :param world_name:
        :return: dump dui2title, dui2desc, dui2idx, idx2dui into ./data/worlds/$world_name/
        '''
        try:
            assert world_name in ALL_WORLDS
        except:
            AssertionError(world_name + "is not included in datasets")

        each_world_dir = self.args.dataset_dir + 'worlds/' + world_name + '/'

        if not os.path.exists(each_world_dir):
            os.mkdir(each_world_dir)

        one_world_json_path = self.args.documents_dir + world_name + '.json'
        one_world_json = oneworld_opener(one_world_json_path)

        dui2title, idx2dui, dui2idx = self.oneworld_json_2_docid2title(one_world_json=one_world_json)
        dui2title = self.dui2tokens_tokentrimmer(dui2tokens=dui2title, max_token=self.args.extracted_first_token_for_title)
        dui2desc_raw = self.oneworld_json_2_dui2desc_raw(one_world_json=one_world_json)
        dui2desc = self.dui2tokens_tokentrimmer(dui2tokens=dui2desc_raw, max_token=self.args.extracted_first_token_for_description)

        jdump(j=dui2title, path=each_world_dir+'dui2title.json')
        jdump(j=idx2dui, path=each_world_dir+'idx2dui.json')
        jdump(j=dui2idx, path=each_world_dir+'dui2idx.json')
        jdump(j=dui2desc, path=each_world_dir+'dui2desc.json')
        jdump(j=dui2desc_raw, path=each_world_dir+'dui2desc_raw.json')

        print('\n==={0} was preprocessed==='.format(world_name))
        print('total entities in {0}:'.format(world_name), len(dui2title))

class MentionParser:
    '''
    dump mention and its gold
    NOTE: Currently, we stand on local model. Also, we only search specified context window
          Because the main goal of this repository is, to confirm bi-encoder effect under zero-shot setting.
          Sentence boundary is not conducted under this class. In future, this must be done.
    '''
    def __init__(self, args):
        self.args = args
        self.flags = ['train', 'test', 'dev']
        self.mention_preprocesseddir_for_each_world_makedir()
        self.worldName_2_dui2rawtext = self.allworlds_loader()

    def train_or_dev_or_test_2_eachworld_splitter(self, train_dev_testflag):
        assert train_dev_testflag in ["train", "dev", "test"]
        mention_itself_path = self.args.mentions_dir + train_dev_testflag + '.json' if train_dev_testflag != "dev" else self.args.mentions_dir + "val" + ".json"

        mention_paths = mentions_in_train_dev_test_loader(mention_itself_path, train_dev_testflag,
                                                          mentions_dir=self.args.mentions_dir,
                                                          worldName_2_dui2rawtext=self.worldName_2_dui2rawtext)
        print('\n{0} mentions are now preprocessed...\n'.format(train_dev_testflag))

        world_2_idx2mention = {}
        skipped = 0
        n_cores = multiprocessing.cpu_count()
        with Pool(n_cores) as pool:
            imap = pool.imap_unordered(mentionConverter, mention_paths)
            result = list(tqdm(imap, total=len(mention_paths)))

        for mention_json_path in tqdm(result):
            # mention_json_path = mentionConverter(one_line_mention_path=mention_path)
            with open(mention_json_path, 'r') as pj:
                mention_json = json.load(pj)
            if int(mention_json['mention_insert_flag']) == 0:
                continue
            # mention_json = self.mentionConverter(one_line_mention_path=mention_path)
            # except:
            #     skipped += 1
            #     print("mention id", mention_path, "is skipped because gold cannot be found")
            #     continue
            world_belongingto = mention_json["gold_world"]
            if world_belongingto not in world_2_idx2mention:
                world_2_idx2mention.update({world_belongingto:{}})
            world_2_idx2mention[world_belongingto].update({len(world_2_idx2mention[world_belongingto]):mention_json})

        for world, its_preprocesseddata in world_2_idx2mention.items():
            jdump(its_preprocesseddata, self.args.mentions_splitbyworld_dir + world + "/mentions.json")


    def mention_preprocesseddir_for_each_world_makedir(self):
        for world in ALL_WORLDS:
            formention_preprocessing_each_world_dir = self.args.mentions_splitbyworld_dir + world + '/'
            if not os.path.exists(formention_preprocessing_each_world_dir):
                os.mkdir(formention_preprocessing_each_world_dir)

    def allworlds_loader(self):
        '''
        :return: allworlds_dui2rawtext.json
        '''
        world_2_dui2rawtext = {}
        for world in ALL_WORLDS:
            path_for_dui2rawtext = self.args.dir_for_each_world + world + '/' + 'dui2desc_raw.json'
            dui2raw = simplejopen(json_file_path=path_for_dui2rawtext)
            world_2_dui2rawtext.update({world:dui2raw})

        return world_2_dui2rawtext


if __name__ == '__main__':
    P = Params()
    opts = P.opts
    t = OneWorldParser(args=opts)
    for world in ALL_WORLDS:
        t.from_oneworld_dump_preprocessed_world(world_name=world)
    mp = MentionParser(args=opts)
    for flag in ["train", "dev", "test"]:
        mp.train_or_dev_or_test_2_eachworld_splitter(train_dev_testflag=flag)


