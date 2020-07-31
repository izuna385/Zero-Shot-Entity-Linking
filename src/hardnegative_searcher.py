from data_reader import OneWorldAllEntityinKBIterateLoader
from utils import oneworld_entiredataset_loader_for_encoding_entities, KBIndexerWithFaiss, BiEncoderTopXRetriever
from utils import parse_duidx2encoded_emb_2_dui2emb
from model import WrappedModel_for_entityencoding
from encoders import InKBAllEntitiesEncoder, BiEncoderForOnlyMentionOutput
import pdb

class HardNegativesSearcherForEachEpochStart:
    def __init__(self, args, world_name, reader, embedder, mention_encoder, entity_encoder,
                 vocab, berttokenizer, bertindexer):

        self.args = args
        self.world_name = world_name
        self.reader = reader
        self.mention_encoder = mention_encoder
        self.entity_encoder = entity_encoder

        self.vocab = vocab
        self.tokenizer = berttokenizer
        self.tokenindexer = bertindexer
        self.embedder = embedder

        # load one world dataset
        self.oneworld_loader()
        self.entity_loader = OneWorldAllEntityinKBIterateLoader(args=self.args, idx2dui=self.idx2dui,
                                                                dui2title=self.dui2title, dui2desc=self.dui2desc,
                                                                textfield_embedder=self.embedder,
                                                                pretrained_tokenizer=self.tokenizer,
                                                                token_indexer=self.tokenindexer)
        self.entity_encoder_wrapping_model = WrappedModel_for_entityencoding(args=args,
                                                                        entity_encoder=self.entity_encoder,
                                                                        vocab=vocab)
        self.encodeAllEntitiesEncoder = InKBAllEntitiesEncoder(args=args,
                                                               entity_loader_datasetreaderclass=self.entity_loader,
                                                               entity_encoder_wrapping_model=self.entity_encoder_wrapping_model,
                                                               vocab=vocab)

    def hardNegativesSearcherandSetter(self, howManySampleSearch=20):
        dui2encoded_emb, duidx2encoded_emb = self.dui2EncoderEntityEmbReturner()
        forstoring_encoded_entities_to_faiss = self.encodedEmbFaissAdder(dui2EncodedEmb=dui2encoded_emb)
        biencoderOnlyForMentionOutput = BiEncoderForOnlyMentionOutput(args=self.args,
                                                                mention_encoder=self.mention_encoder,
                                                                vocab=self.vocab)

        biencoderOnlyForMentionOutput.cuda(), biencoderOnlyForMentionOutput.eval()

        retriever = self.topXRetieverLoader(biencoderOnlyForMentionOutputClass=biencoderOnlyForMentionOutput,
                                            forstoring_encoded_entities_to_faissAdderclass=forstoring_encoded_entities_to_faiss,
                                            duidx2encoded_emb=duidx2encoded_emb)

        mentionId2GoldDUIDX, mentionId2HardNegativeDUIDX = {}, {}
        print("\n########\nHARD NEGATIVE MININGS started\n########\n")
        for faiss_search_candidate_result_duidxs, mention_uniq_ids, gold_duidxs in retriever.biencoder_tophits_retrievaler(train_or_dev_or_test_flag='train',
                                                                                                                           how_many_top_hits_preserved=howManySampleSearch):
            for one_mention_search_result, mention_uniq_id, gold_duidx in zip(faiss_search_candidate_result_duidxs, mention_uniq_ids, gold_duidxs):
                hard_negatives = one_mention_search_result[one_mention_search_result != gold_duidx][:self.args.hard_negatives_num].tolist()
                mention_uniq_id = int(mention_uniq_id)
                gold_duidx = int(gold_duidx)
                mentionId2GoldDUIDX.update({mention_uniq_id:gold_duidx})
                mentionId2HardNegativeDUIDX.update({mention_uniq_id:hard_negatives})

        self.reader.hardNegativesUpdater(mentionId2GoldDUIDX=mentionId2GoldDUIDX, mentionId2HardNegativeDUIDX=mentionId2HardNegativeDUIDX)

    def topXRetieverLoader(self, biencoderOnlyForMentionOutputClass, forstoring_encoded_entities_to_faissAdderclass,
                           duidx2encoded_emb):

        return BiEncoderTopXRetriever(args=self.args,vocab=self.vocab,
                                      biencoder_onlyfor_encodingmentions=biencoderOnlyForMentionOutputClass,
                                      faiss_stored_kb=forstoring_encoded_entities_to_faissAdderclass.indexed_faiss_returner(),
                                      reader_for_mentions=self.reader,
                                      duidx2encoded_emb=duidx2encoded_emb)

    def dui2EncoderEntityEmbReturner(self):
        duidx2encoded_emb = self.encodeAllEntitiesEncoder.encoding_all_entities()
        dui2encoded_emb = parse_duidx2encoded_emb_2_dui2emb(duidx2encoded_emb=duidx2encoded_emb, original_dui2idx=self.dui2idx)

        return dui2encoded_emb, duidx2encoded_emb

    def encodedEmbFaissAdder(self, dui2EncodedEmb):
        return KBIndexerWithFaiss(args=self.args, input_dui2idx=self.dui2idx, input_idx2dui=self.idx2dui,
                                  input_dui2emb=dui2EncodedEmb, search_method_for_faiss=self.args.search_method,
                                  entity_emb_dim=self.entityEmbDimReturner())

    def oneworld_loader(self):
        '''
        load self.dui2desc, self.dui2title, self.idx2dui
        :return:
        '''
        self.dui2idx, self.idx2dui, self.dui2title, self.dui2desc = oneworld_entiredataset_loader_for_encoding_entities(args=self.args,
                                                                                                                        world_name=self.world_name)
    def entityEmbDimReturner(self):
        if self.args.dimentionReduction:
            return self.args.dimentionReductionToThisDim
        else:
            return 768
