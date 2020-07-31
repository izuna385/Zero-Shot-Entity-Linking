'''
Model classes
'''
import torch
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import normalize
import torch.nn.functional as F
import copy
import pdb

class Biencoder(Model):
    def __init__(self, args,
                 mention_encoder: Seq2VecEncoder,
                 entity_encoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.mention_encoder = mention_encoder
        self.accuracy = CategoricalAccuracy()
        self.istrainflag = 1
        self.BCEWloss = nn.BCEWithLogitsLoss(reduction="mean")
        self.mesloss = nn.MSELoss()
        self.entity_encoder = entity_encoder
        self.cuda_flag = 1

    def forward(self, context, gold_title_and_desc_concatenated, gold_and_negatives_title_and_desc_concatenated,
                gold_duidx, mention_uniq_id, labels_for_BCEWithLogitsLoss):
        batch_num = context['tokens'].size(0)
        contextualized_mention = self.mention_encoder(context)
        encoded_entites = self.entity_encoder(title_and_desc_concatnated_text=gold_title_and_desc_concatenated)

        contextualized_mention_forcossim = normalize(contextualized_mention, dim=1)
        if self.args.search_method == 'cossim':
            encoded_entites_forcossim = normalize(encoded_entites, dim=1)
            scores = contextualized_mention_forcossim.mm(encoded_entites_forcossim.t())
        elif self.args.search_method == 'indexflatip':
            scores = contextualized_mention.mm(encoded_entites.t())
        else:
            assert self.args.search_method == 'indexflatl2'
            scores = - self.calc_L2distance(contextualized_mention.view(batch_num, 1, -1), encoded_entites) # FIXED

        device = torch.get_device(scores) if self.cuda_flag else torch.device('cpu')
        target = torch.LongTensor(torch.arange(batch_num)).to(device)

        if self.args.search_method in ['cossim','indexflatip']:
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss = self.BCEWloss(scores, torch.eye(batch_num).cuda())

        output = {'loss': loss}

        if self.args.add_mse_for_biencoder:
            output['loss'] += self.mesloss(contextualized_mention, encoded_entites)

        if self.args.add_hard_negatives:
            batch_, gold_plus_negs_num, maxTokenLengthInBatch = gold_and_negatives_title_and_desc_concatenated['tokens'].size()
            docked_tokenlist = {'tokens': gold_and_negatives_title_and_desc_concatenated['tokens'].view(batch_ * gold_plus_negs_num, -1)}
            encoded_entities_from_hard_negatives_idx0isgold = self.entity_encoder(docked_tokenlist).view(batch_, gold_plus_negs_num, -1)

            if self.args.search_method == 'cossim':
                encoded_mention_forcossim_ = contextualized_mention_forcossim.repeat(1, gold_plus_negs_num).view(batch_*gold_plus_negs_num, -1)
                scores_for_hard_negatives = (((encoded_mention_forcossim_)*(normalize(encoded_entities_from_hard_negatives_idx0isgold.view(batch_*gold_plus_negs_num,-1), dim=1))).sum(1, keepdim=True).squeeze(1)).view(batch_,gold_plus_negs_num)
            elif self.args.search_method == 'indexflatip':
                encoded_mention_ = contextualized_mention.repeat(1, gold_plus_negs_num).view(batch_* gold_plus_negs_num, -1)
                scores_for_hard_negatives = (((encoded_mention_)*(encoded_entities_from_hard_negatives_idx0isgold.view(batch_*gold_plus_negs_num,-1))).sum(1, keepdim=True).squeeze(1)).view(batch_,gold_plus_negs_num)
            else:
                raise NotImplementedError

            loss += self.BCEWloss(scores_for_hard_negatives, labels_for_BCEWithLogitsLoss)

        if self.istrainflag:
            golds = torch.eye(batch_num).to(device)
            self.accuracy(scores, torch.argmax(golds, dim=1))
        else:
            output['gold_duidx'] = gold_duidx
            output['encoded_mentions'] = contextualized_mention

        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def return_entity_encoder(self):
        return self.entity_encoder

    def switch2eval(self):
        self.istrainflag = copy.copy(0)

    def calc_L2distance(self, h, t):
        diff = h - t
        return torch.norm(diff, dim=2)  # batch * cands
        # encoded_entites.unsqueeze(0).repeat(batch_num,1,1)

class WrappedModel_for_entityencoding(Model):
    def __init__(self, args,
                 entity_encoder,
                 vocab):
        super().__init__(vocab)
        self.args = args
        self.entity_encoder = entity_encoder

    def forward(self, duidx, title_and_desc_concatnated_text):

        encoded_entites = self.entity_encoder(title_and_desc_concatnated_text=title_and_desc_concatnated_text)
        output = {'gold_duidx': duidx, 'emb_of_entities_encoded': encoded_entites}

        return output

    def return_entity_encoder(self):
        return self.entity_encoder
