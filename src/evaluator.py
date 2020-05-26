from data_reader import OneWorldAllEntityinKBIterateLoader
from utils import simplejopen, j_intidx2str_opener, parse_duidx2encoded_emb_for_debugging, parse_duidx2encoded_emb_2_dui2emb, KBIndexerWithFaiss, jdump
from utils import oneworld_entiredataset_loader_for_encoding_entities, BiEncoderTopXRetriever
from model import WrappedModel_for_entityencoding
from encoders import InKBAllEntitiesEncoder, BiEncoderForOnlyMentionOutput
from utils_for_evaluator import DevandTest_BiEncoder_IterateEvaluator
import pdb, os

class Evaluate_one_world:
    def __init__(self, args, world_name, reader, embedder, trainfinished_mention_encoder,trainfinished_entity_encoder,
                 vocab, experiment_logdir, dev_or_test, berttokenizer, bertindexer):
        self.args = args
        self.world_name = world_name
        self.reader = reader
        self.trainfinished_entity_encoder = trainfinished_entity_encoder
        self.trainfinished_mention_encoder = trainfinished_mention_encoder
        self.vocab = vocab
        self.experiment_logdir = experiment_logdir
        self.dev_or_test = dev_or_test

        self.tokenizer = berttokenizer
        self.tokenindexer = bertindexer
        self.embedder = embedder

        print('===loading world {0}'.format(world_name))
        self.oneworld_loader()
        self.entity_loader = OneWorldAllEntityinKBIterateLoader(args=self.args, idx2dui=self.idx2dui, dui2title=self.dui2title, dui2desc=self.dui2title,
                 textfield_embedder=self.embedder, pretrained_tokenizer=self.tokenizer, token_indexer=self.tokenindexer)
        print('===world loaded!===')

        self.entity_encoder_wrapping_model = WrappedModel_for_entityencoding(args=args,
                                                                        entity_encoder=self.trainfinished_entity_encoder,
                                                                        vocab=vocab)
        self.encodeAllEntitiesEncoder = InKBAllEntitiesEncoder(args=args,
                                                               entity_loader_datasetreaderclass=self.entity_loader,
                                                               entity_encoder_wrapping_model=self.entity_encoder_wrapping_model,
                                                               vocab=vocab)

    def evaluate_one_world(self):
        dui2encoded_emb =  self.dui2EncoderEntityEmbReturner()
        print('=====Encoding all entities in KB FINISHED!=====')

        print('\n+++++Indexnizing KB from encoded entites+++++')
        forstoring_encoded_entities_to_faiss = self.encodedEmbFaissAdder(dui2EncodedEmb=dui2encoded_emb)
        print('+++++Indexnizing KB from encoded entites FINISHED!+++++')

        print('Loading Biencoder')
        biencoder_onlyfor_encodingmentions = BiEncoderForOnlyMentionOutput(args=self.args,
                                                                mention_encoder=self.trainfinished_mention_encoder,
                                                                vocab=self.vocab)
        biencoder_onlyfor_encodingmentions.cuda()
        biencoder_onlyfor_encodingmentions.eval()
        print('Loaded: Biencoder')

        print('Evaluation for BiEncoder start')
        topXretriever = BiEncoderTopXRetriever(args=self.args,
                                               vocab=self.vocab,
                                               biencoder_onlyfor_encodingmentions=biencoder_onlyfor_encodingmentions,
                                               faiss_stored_kb=forstoring_encoded_entities_to_faiss.indexed_faiss_returner(),
                                               reader_for_mentions=self.reader)

        oneworld_evaluator = DevandTest_BiEncoder_IterateEvaluator(args=self.args,
                                                                   BiEncoderEvaluator=topXretriever,
                                                                   experiment_logdir=self.experiment_logdir,
                                                                   world_name=self.world_name)
        Hits1count, Hits10count, Hits50count, Hits100count, Hits500count, data_points = oneworld_evaluator.final_evaluation(dev_or_test_flag=self.dev_or_test)

        self.log_one_world(h1count=Hits1count, h10count=Hits10count, h50count=Hits50count, h100count=Hits100count, h500count=Hits500count, data_points=data_points)
        return Hits1count, Hits10count, Hits50count, Hits100count, Hits500count, data_points

    def log_one_world(self, h1count, h10count, h50count, h100count, h500count, data_points):
        if not os.path.exists(self.experiment_logdir + self.dev_or_test):
            os.mkdir(self.experiment_logdir + self.dev_or_test)
        dumped_jsonpath = self.experiment_logdir + self.dev_or_test + '/' + self.world_name + 'eval.json'
        jdump(j={'h1_percent': h1count / data_points * 100,
                 'h10_percent': h10count / data_points * 100,
                 'h50_percent': h50count / data_points * 100,
                 'h100_percent': h100count / data_points * 100,
                 'h500_percent': h500count / data_points * 100,
                 'data_points':data_points
                 }, path=dumped_jsonpath)


    def oneworld_loader(self):
        '''
        load self.dui2desc, self.dui2title, self.idx2dui
        :return:
        '''
        self.dui2idx, self.idx2dui, self.dui2title, self.dui2desc = oneworld_entiredataset_loader_for_encoding_entities(args=self.args,
                                                                                                                        world_name=self.world_name)

    def duidx2EncodedEmbReturner(self):
        return self.encodeAllEntitiesEncoder.encoding_all_entities()

    def dui2EncoderEntityEmbReturner(self):
        duidx2encoded_emb = self.encodeAllEntitiesEncoder.encoding_all_entities()
        dui2encoded_emb = parse_duidx2encoded_emb_2_dui2emb(duidx2encoded_emb=duidx2encoded_emb,
                                                            original_dui2idx=self.dui2idx)

        return dui2encoded_emb

    def encodedEmbFaissAdder(self, dui2EncodedEmb):
        return KBIndexerWithFaiss(args=self.args, input_dui2idx=self.dui2idx,
                                  input_idx2dui=self.idx2dui, input_dui2emb=dui2EncodedEmb,
                                  search_method_for_faiss=self.args.search_method,
                                  entity_emb_dim=768)