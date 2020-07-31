import json, os, sys, pdb
from datetime import datetime
from pytz import timezone
from allennlp.common.tee_logger import TeeLogger
import numpy as np
from tqdm import tqdm
import faiss
from commons import TRAIN_WORLDS, DEV_WORLDS, TEST_WORLDS
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
from tqdm import tqdm
import torch
from torch.nn.functional import normalize

def dummynegativesloader(mentionNumbers=100):
    mention_uniq_id2negatives = {}
    dummy_negative_dui_idxs = [0, 1]
    for mention_uniq_id in range(mentionNumbers):
        mention_uniq_id2negatives.update({mention_uniq_id:dummy_negative_dui_idxs})

    return mention_uniq_id2negatives

def worlds_loader(args):
    if args.debug:
        return ["yugioh"], ["yugioh"],  ["yugioh"]
    else:
        return TRAIN_WORLDS, DEV_WORLDS, TEST_WORLDS

def oneworld_entiredataset_loader_for_encoding_entities(args, world_name):
    '''
    load self.dui2desc, self.dui2title, self.idx2dui
    :return:
    '''
    worlds_dir = args.dir_for_each_world + world_name + '/'
    dui2desc_path = worlds_dir + "dui2desc.json"
    dui2title_path = worlds_dir + "dui2title.json"
    idx2dui_path = worlds_dir + "idx2dui.json"

    dui2desc = simplejopen(dui2desc_path)
    dui2title = simplejopen(dui2title_path)
    idx2dui = j_intidx2str_opener(idx2dui_path)

    dui2idx = {}
    for idx, dui in idx2dui.items():
        dui2idx.update({dui:int(idx)})

    return dui2idx, idx2dui, dui2title, dui2desc

def simplejopen(json_file_path):
    with open(json_file_path, 'r') as f:
        j = json.load(f)
    return j

def j_str2intidx_opener(json_file_path):
    with open(json_file_path, 'r') as f:
        j = json.load(f)
    st2intidx = {}
    for k, stidx in j.items():
        st2intidx.update({k:int(stidx)})

    return st2intidx

def j_intidx2str_opener(json_file_path):
    with open(json_file_path, 'r') as f:
        j = json.load(f)
    intidx2v = {}
    for stidx, v in j.items():
        intidx2v.update({int(stidx):v})

    return intidx2v

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

def dev_or_test_finallog(entire_h1c, entire_h10c, entire_h50c, entire_h64c, entire_h100c, entire_h500c,entire_datapoints, dev_or_test_flag, experiment_logdir):
    jpath = experiment_logdir + dev_or_test_flag + '_eval.json'
    jdump(path=jpath,
          j={
              'entire_h1_percent': entire_h1c / entire_datapoints * 100,
              'entire_h10_percent': entire_h10c / entire_datapoints * 100,
              'entire_h50_percent': entire_h50c / entire_datapoints * 100,
              'entire_h64_percent': entire_h64c / entire_datapoints * 100,
              'entire_h100_percent': entire_h100c / entire_datapoints * 100,
              'entire_h500_percent': entire_h500c / entire_datapoints * 100
          })

def experiment_logger(args):
    '''
    :param args: from biencoder_parameters
    :return: dirs for experiment log
    '''
    experimet_logdir = args.experiment_logdir # / is included
    if not os.path.exists(experimet_logdir):
        os.mkdir(experimet_logdir)

    timestamp = datetime.now(timezone('Asia/Tokyo'))
    str_timestamp = '{0:%Y%m%d_%H%M%S}'.format(timestamp)[2:]

    dir_for_each_experiment = experimet_logdir + str_timestamp

    if os.path.exists(dir_for_each_experiment):
        dir_for_each_experiment += '_d'

    dir_for_each_experiment += '/'
    logger_path = dir_for_each_experiment + 'teelog.log'
    os.mkdir(dir_for_each_experiment)

    if not args.debug:
        sys.stdout = TeeLogger(logger_path, sys.stdout, False)  # default: False
        sys.stderr = TeeLogger(logger_path, sys.stderr, False)  # default: False

    return dir_for_each_experiment

def cuda_device_parser(str_ids):
    return [int(stridx) for stridx in str_ids.strip().split(',')]

def parse_duidx2encoded_emb_for_debugging(duidx2encoded_emb, original_dui2idx):
    print('/////Some entities embs are randomized for debugging./////')
    for duidx in tqdm(original_dui2idx.values()):
        if duidx not in duidx2encoded_emb:
            duidx2encoded_emb.update({duidx:np.random.randn(*duidx2encoded_emb[0].shape)})
    return duidx2encoded_emb

def parse_duidx2encoded_emb_2_dui2emb(duidx2encoded_emb, original_dui2idx):
    dui2emb = {}
    for dui, idx in original_dui2idx.items():
        dui2emb.update({dui:duidx2encoded_emb[idx]})
    return dui2emb


class KBIndexerWithFaiss:
    def __init__(self, args, input_dui2idx, input_idx2dui, input_dui2emb, search_method_for_faiss, entity_emb_dim=300):
        self.args = args
        self.kbemb_dim = entity_emb_dim
        self.dui2idx = input_dui2idx
        self.idx2dui = input_idx2dui
        self.dui2emb = input_dui2emb
        self.search_method_for_faiss = search_method_for_faiss
        self.indexed_faiss_loader()
        self.KBmatrix = self.KBmatrixloader()
        self.entity_num = len(input_dui2idx)
        self.indexed_faiss_KBemb_adder(KBmatrix=self.KBmatrix)

    def KBmatrixloader(self):
        KBemb = np.random.randn(len(self.dui2idx.keys()), self.kbemb_dim).astype('float32')
        for idx, dui in self.idx2dui.items():
            KBemb[idx] = self.dui2emb[dui]

        return KBemb

    def indexed_faiss_loader(self):
        if self.search_method_for_faiss == 'indexflatl2':  # L2
            self.indexed_faiss = faiss.IndexFlatL2(self.kbemb_dim)
        elif self.search_method_for_faiss == 'indexflatip':  #
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)
        elif self.search_method_for_faiss == 'cossim':  # innerdot * Beforehand-Normalization must be done.
            self.indexed_faiss = faiss.IndexFlatIP(self.kbemb_dim)

    def indexed_faiss_KBemb_adder(self, KBmatrix):
        if self.search_method_for_faiss == 'cossim':
            KBemb_normalized_for_cossimonly = np.random.randn(self.entity_num, self.kbemb_dim).astype('float32')
            for idx, emb in enumerate(KBmatrix):
                if np.linalg.norm(emb, ord=2, axis=0) != 0:
                    KBemb_normalized_for_cossimonly[idx] = emb / np.linalg.norm(emb, ord=2, axis=0)
            self.indexed_faiss.add(KBemb_normalized_for_cossimonly)
        else:
            self.indexed_faiss.add(KBmatrix)

    def indexed_faiss_returner(self):
        return self.indexed_faiss

class BiEncoderTopXRetriever:
    def __init__(self, args, vocab, biencoder_onlyfor_encodingmentions, faiss_stored_kb, reader_for_mentions,
                 duidx2encoded_emb):
        self.args = args
        self.mention_encoder = biencoder_onlyfor_encodingmentions
        self.mention_encoder.eval()
        self.faiss_searcher = faiss_stored_kb
        self.reader_for_mentions = reader_for_mentions
        self.sequence_iterator = BasicIterator(batch_size=self.args.batch_size_for_eval)
        self.sequence_iterator.index_with(vocab)
        self.cuda_device = 0
        self.duidx2encoded_emb = duidx2encoded_emb

    def biencoder_tophits_retrievaler(self, train_or_dev_or_test_flag, how_many_top_hits_preserved=500):
        ds = self.reader_for_mentions.read(train_or_dev_or_test_flag)
        generator_for_biencoder = self.sequence_iterator(ds, num_epochs=1, shuffle=False)
        generator_for_biencoder_tqdm = tqdm(generator_for_biencoder, total=self.sequence_iterator.get_num_batches(ds))

        with torch.no_grad():
            for batch in generator_for_biencoder_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                mention_uniq_ids, encoded_mentions, gold_duidxs = self._extract_mention_idx_encoded_emb_and_its_gold_cuidx(batch=batch)
                faiss_search_candidate_result_cuidxs = self.faiss_topx_retriever(encoded_mentions=encoded_mentions,
                                                                                 how_many_top_hits_preserved=how_many_top_hits_preserved)
                yield faiss_search_candidate_result_cuidxs, mention_uniq_ids, gold_duidxs

    def faiss_topx_retriever(self, encoded_mentions, how_many_top_hits_preserved):
        '''
        if cossimsearch -> re-sort with L2, we have to use self.args.cand_num_before_sort_candidates_forBLINKbiencoder
        Args:
            encoded_mentions:
            how_many_top_hits_preserved:
        Returns:
        '''

        if self.args.search_method == 'cossim':
            encoded_mentions = normalize(torch.from_numpy(encoded_mentions), dim=1).cpu().detach().numpy()
            _, faiss_search_candidate_result_cuidxs = self.faiss_searcher.search(encoded_mentions, how_many_top_hits_preserved)

        else:
            # assert self.args.search_method == 'indexflatl2'
            _, faiss_search_candidate_result_cuidxs = self.faiss_searcher.search(encoded_mentions, how_many_top_hits_preserved)

        return faiss_search_candidate_result_cuidxs

    def calc_L2distance(self, h, t):
        diff = h - t
        return torch.norm(diff, dim=2)

    def tonp(self, tsr):
        return tsr.detach().cpu().numpy()

    def _extract_mention_idx_encoded_emb_and_its_gold_cuidx(self, batch):
        out_dict = self.mention_encoder(**batch)
        return self.tonp(out_dict['mention_uniq_id']), self.tonp(out_dict['contextualized_mention']), self.tonp(out_dict['gold_duidx'])