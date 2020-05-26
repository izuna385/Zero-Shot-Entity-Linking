import pdb
from allennlp.nn import util as nn_util
from allennlp.data.iterators import BasicIterator
from tqdm import tqdm
import torch
from torch.nn.functional import normalize
import numpy as np
import math, json

class DevandTest_BiEncoder_IterateEvaluator:
    def __init__(self, args, BiEncoderEvaluator, experiment_logdir, world_name):
        self.BiEncoderEvaluator = BiEncoderEvaluator
        self.experiment_logdir = experiment_logdir
        self.args = args
        self.world_name = world_name

    # TODO: Add L2 based sorting
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
