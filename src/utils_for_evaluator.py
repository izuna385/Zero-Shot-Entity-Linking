import pdb
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
from tqdm import tqdm
import torch
from torch.nn.functional import normalize
import numpy as np
import math, json

class BiEncoderTopXRetriever:
    def __init__(self, args, vocab, biencoder_onlyfor_encodingmentions, faiss_stored_kb, reader_for_mentions):
        self.args = args
        self.mention_encoder = biencoder_onlyfor_encodingmentions
        self.mention_encoder.eval()
        self.faiss_searcher = faiss_stored_kb
        self.reader_for_mentions = reader_for_mentions
        self.sequence_iterator = BasicIterator(batch_size=self.args.batch_size_for_eval)
        self.sequence_iterator.index_with(vocab)
        self.cuda_device = 0

    def biencoder_tophits_retrievaler(self, dev_or_test_flag, how_many_top_hits_preserved=500):
        ds = self.reader_for_mentions.read(dev_or_test_flag)
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

    def _extract_mention_idx_encoded_emb_and_its_gold_cuidx(self, batch) -> np.ndarray:
        out_dict = self.mention_encoder(**batch)
        return self.tonp(out_dict['mention_uniq_id']), self.tonp(out_dict['contextualized_mention']), self.tonp(out_dict['gold_duidx'])

class DevandTest_BiEncoder_IterateEvaluator:
    def __init__(self, args, BiEncoderEvaluator, experiment_logdir, world_name):
        self.BiEncoderEvaluator = BiEncoderEvaluator
        self.experiment_logdir = experiment_logdir
        self.args = args
        self.world_name = world_name

    def final_evaluation(self, dev_or_test_flag, how_many_top_hits_preserved=500):
        print('============\n<<<FINAL EVALUATION STARTS>>>', self.world_name, 'in', dev_or_test_flag, 'Retrieve_Candidates:', how_many_top_hits_preserved,'\n============')
        Hits1, Hits10, Hits50, Hits100, Hits500 = 0, 0, 0, 0, 0
        data_points = 0

        for faiss_search_candidate_result_duidxs, mention_uniq_ids, gold_duidxs in self.BiEncoderEvaluator.biencoder_tophits_retrievaler(dev_or_test_flag, how_many_top_hits_preserved):
            b_Hits1, b_Hits10, b_Hits50, b_Hits100, b_Hits500 = self.batch_candidates_and_gold_cuiddx_2_batch_hits(faiss_search_candidate_result_duidxs=faiss_search_candidate_result_duidxs,
                                                                                                        gold_duidxs=gold_duidxs)
            assert len(mention_uniq_ids) == len(gold_duidxs)
            data_points += len(mention_uniq_ids)
            Hits1 += b_Hits1
            Hits10 += b_Hits10
            Hits50 += b_Hits50
            Hits100 += b_Hits100
            Hits500 += b_Hits500

        return Hits1, Hits10, Hits50, Hits100, Hits500, data_points

    def batch_candidates_and_gold_cuiddx_2_batch_hits(self, faiss_search_candidate_result_duidxs, gold_duidxs):
        b_Hits1, b_Hits10, b_Hits50, b_Hits100, b_Hits500 = 0, 0, 0, 0, 0
        for candidates_sorted, gold_idx in zip(faiss_search_candidate_result_duidxs, gold_duidxs):
            if len(np.where(candidates_sorted == int(gold_idx))[0]) != 0:
                rank = int(np.where(candidates_sorted == int(gold_idx))[0][0])

                if rank == 0:
                    b_Hits1 += 1
                    b_Hits10 += 1
                    b_Hits50 += 1
                    b_Hits100 += 1
                    b_Hits500 += 1
                elif rank < 10:
                    b_Hits10 += 1
                    b_Hits50 += 1
                    b_Hits100 += 1
                    b_Hits500 += 1
                elif rank < 50:
                    b_Hits50 += 1
                    b_Hits100 += 1
                    b_Hits500 += 1
                elif rank < 100:
                    b_Hits100 += 1
                    b_Hits500 += 1
                elif rank < 500:
                    b_Hits500 += 1
                else:
                    continue

        return b_Hits1, b_Hits10, b_Hits50, b_Hits100, b_Hits500
