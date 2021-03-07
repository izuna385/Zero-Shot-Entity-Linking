'''
CREATE dui2cano, dui2def from documents and iterative mentions
dui: Document ID in each world.
'''
from commons import TRAIN_WORLDS, DEV_WORLDS, TEST_WORLDS, MENTION_START_TOKEN, MENTION_END_TOKEN
import os, pdb
from tqdm import tqdm
import json
from parameters import Params
ALL_WORLDS = TRAIN_WORLDS + DEV_WORLDS + TEST_WORLDS

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

def mentions_in_train_dev_test_loader(mention_jsonpath):
    lines = []
    with open(mention_jsonpath, 'r') as f:
        for line in f:
            json_parsed = json.loads(line)
            lines.append(json_parsed)

    return lines

def jdump(j, path):
    with open(path, 'w') as f:
        json.dump(j, f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

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
        mention_path = self.args.mentions_dir + train_dev_testflag + '.json' if train_dev_testflag != "dev" else self.args.mentions_dir + "val" + ".json"
        mentions = mentions_in_train_dev_test_loader(mention_path)
        print('\n{0} mentions are now preprocessed...\n'.format(train_dev_testflag))

        world_2_idx2mention = {}
        skipped = 0
        for mention_data in tqdm(mentions):
            try:
                mention_json = self.mentionConverter(one_line_mention=mention_data)
            except:
                skipped += 1
                print("mention id", mention_data["mention_id"], "is skipped because gold cannot be found")
                continue
            world_belongingto = mention_json["gold_world"]
            if world_belongingto not in world_2_idx2mention:
                world_2_idx2mention.update({world_belongingto:{}})
            world_2_idx2mention[world_belongingto].update({len(world_2_idx2mention[world_belongingto]):mention_json})

        for world, its_preprocesseddata in world_2_idx2mention.items():
            jdump(its_preprocesseddata, self.args.mentions_splitbyworld_dir + world + "/mentions.json")

    def mentionConverter(self, one_line_mention):
        '''
        :param one_line_mention:
        :return: labeled_dui, mention itself, context wrapped with start/end token
        '''
        # one_line_mention = {"category": "MULTIPLE_CATEGORIES", "text": "I - Spy Express", "end_index": 842, "context_document_id": "8D5C15C034E982AF", "label_document_id": "0ABE561452932284", "mention_id": "FFF2DE1A167361C1", "corpus": "lego", "start_index": 839}
        raw_mention = one_line_mention["text"]
        gold_dui = one_line_mention["label_document_id"]
        dui_where_mention_exist = one_line_mention["context_document_id"]
        gold_world = one_line_mention["corpus"]

        mention_start_tokenidx = one_line_mention["start_index"]
        mention_end_tokenidx = one_line_mention["end_index"]

        # TODO: SENTENCE BOUNDARY DETECTION
        context_start_index, context_end_index = 0, 0
        if mention_start_tokenidx - self.args.mention_leftandright_tokenwindowwidth >= 0:
            context_start_index += mention_start_tokenidx - self.args.mention_leftandright_tokenwindowwidth
            context_end_index += mention_end_tokenidx + self.args.mention_leftandright_tokenwindowwidth
        else:
            context_end_index = self.args.mention_leftandright_tokenwindowwidth * 2 - mention_end_tokenidx

        raw_context = self.worldName_2_dui2rawtext[gold_world][dui_where_mention_exist].strip().split(' ')
        raw_context.insert(mention_start_tokenidx, MENTION_START_TOKEN)
        raw_context.insert(mention_end_tokenidx + 2, MENTION_END_TOKEN)

        context = raw_context[context_start_index:context_end_index]

        return {"raw_mention": raw_mention,
                "gold_dui": gold_dui,
                "gold_world": gold_world,
                "anchored_context": context}

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


