from data_reader import OneWorldAllEntityinKBIterateLoader
from utils import parse_duidx2encoded_emb_2_dui2emb, KBIndexerWithFaiss, jdump
from utils import oneworld_entiredataset_loader_for_encoding_entities, BiEncoderTopXRetriever
from model import WrappedModel_for_entityencoding
from encoders import InKBAllEntitiesEncoder, BiEncoderForOnlyMentionOutput
from utils_for_evaluator import DevandTest_BiEncoder_IterateEvaluator
import pdb, os
from commons import DEV_WORLDS, TEST_WORLDS, DevEvalDuringTrainDirForEachExperiment
import copy
from data_reader import WorldsReader

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
        self.entity_loader = OneWorldAllEntityinKBIterateLoader(args=self.args, idx2dui=self.idx2dui,
                                                                dui2title=self.dui2title, dui2desc=self.dui2title,
                                                                textfield_embedder=self.embedder,
                                                                pretrained_tokenizer=self.tokenizer,
                                                                token_indexer=self.tokenindexer)
        print('===world loaded!===')

        self.entity_encoder_wrapping_model = WrappedModel_for_entityencoding(args=args,
                                                                        entity_encoder=self.trainfinished_entity_encoder,
                                                                        vocab=vocab)
        self.entity_encoder_wrapping_model.eval()

        self.encodeAllEntitiesEncoder = InKBAllEntitiesEncoder(args=args,
                                                               entity_loader_datasetreaderclass=self.entity_loader,
                                                               entity_encoder_wrapping_model=self.entity_encoder_wrapping_model,
                                                               vocab=vocab)

    def evaluate_one_world(self, trainEpoch=-1):
        dui2encoded_emb, duidx2encoded_emb = self.dui2EncoderEntityEmbReturner()
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
                                               reader_for_mentions=self.reader,
                                               duidx2encoded_emb=duidx2encoded_emb
                                               )

        oneworld_evaluator = DevandTest_BiEncoder_IterateEvaluator(args=self.args,
                                                                   BiEncoderEvaluator=topXretriever,
                                                                   experiment_logdir=self.experiment_logdir,
                                                                   world_name=self.world_name)

        Hits1count, Hits10count, Hits50count, Hits64count, Hits100count, Hits500count, data_points = oneworld_evaluator.final_evaluation(train_or_dev_or_test_flag=self.dev_or_test)

        if trainEpoch == -1:
            self.log_one_world(h1count=Hits1count, h10count=Hits10count, h50count=Hits50count, h64count=Hits64count,
                               h100count=Hits100count, h500count=Hits500count, data_points=data_points)
        else:
            self.logDevEvaluationOfOneWorldDuringTrain(h1count=Hits1count, h10count=Hits10count,
                                                       h50count=Hits50count, h64count=Hits64count,
                                                       h100count=Hits100count, h500count=Hits500count,
                                                       data_points=data_points, trainEpoch=trainEpoch)

        return Hits1count, Hits10count, Hits50count, Hits64count, Hits100count, Hits500count, data_points

    def log_one_world(self, h1count, h10count, h50count, h64count, h100count, h500count, data_points):
        if not os.path.exists(self.experiment_logdir + 'final_' + self.dev_or_test):
            os.mkdir(self.experiment_logdir + 'final_' + self.dev_or_test)
        dumped_jsonpath = self.experiment_logdir + 'final_' + self.dev_or_test + '/' + self.world_name + '_eval.json'

        jdump(j={'h1_percent': h1count / data_points * 100,
                 'h10_percent': h10count / data_points * 100,
                 'h50_percent': h50count / data_points * 100,
                 'h64_percent': h64count / data_points * 100,
                 'h100_percent': h100count / data_points * 100,
                 'h500_percent': h500count / data_points * 100,
                 'data_points':data_points
                 }, path=dumped_jsonpath)

    def logDevEvaluationOfOneWorldDuringTrain(self, h1count, h10count, h50count, h64count, h100count, h500count,
                                              data_points, trainEpoch):
        if not os.path.exists(self.experiment_logdir + DevEvalDuringTrainDirForEachExperiment + '/'):
            os.mkdir(self.experiment_logdir + DevEvalDuringTrainDirForEachExperiment + '/')
        if not os.path.exists(self.experiment_logdir + DevEvalDuringTrainDirForEachExperiment + '/' + self.world_name + '/'):
            os.mkdir(self.experiment_logdir + DevEvalDuringTrainDirForEachExperiment + '/' + self.world_name + '/')
        dumped_jsonpath = self.experiment_logdir + DevEvalDuringTrainDirForEachExperiment + '/' + self.world_name + '/' + 'devEval_ep' + str(trainEpoch) + '.json'

        jdump(j={'h1_percent': h1count / data_points * 100,
                 'h10_percent': h10count / data_points * 100,
                 'h50_percent': h50count / data_points * 100,
                 'h64_percent': h64count / data_points * 100,
                 'h100_percent': h100count / data_points * 100,
                 'h500_percent': h500count / data_points * 100,
                 'data_points': data_points,
                 'world_name': self.world_name,
                 'train_ep' : trainEpoch
                 }, path=dumped_jsonpath)


    def oneworld_loader(self):
        '''
        load self.dui2desc, self.dui2title, self.idx2dui
        :return:
        '''
        self.dui2idx, self.idx2dui, self.dui2title, self.dui2desc = \
            oneworld_entiredataset_loader_for_encoding_entities(args=self.args, world_name=self.world_name)

    def duidx2EncodedEmbReturner(self):
        return self.encodeAllEntitiesEncoder.encoding_all_entities()

    def dui2EncoderEntityEmbReturner(self):
        duidx2encoded_emb = self.encodeAllEntitiesEncoder.encoding_all_entities()
        dui2encoded_emb = parse_duidx2encoded_emb_2_dui2emb(duidx2encoded_emb=duidx2encoded_emb,
                                                            original_dui2idx=self.dui2idx)

        return dui2encoded_emb, duidx2encoded_emb

    def encodedEmbFaissAdder(self, dui2EncodedEmb):
        return KBIndexerWithFaiss(args=self.args, input_dui2idx=self.dui2idx,
                                  input_idx2dui=self.idx2dui, input_dui2emb=dui2EncodedEmb,
                                  search_method_for_faiss=self.args.search_method,
                                  entity_emb_dim=self.entityEmbDimReturner())

    def entityEmbDimReturner(self):
        if self.args.dimentionReduction:
            return self.args.dimentionReductionToThisDim
        else:
            return 768

def oneLineLoaderForDevOrTestEvaluation(dev_or_test_flag, opts, global_tokenIndexer, global_tokenizer,
                                        textfieldEmbedder, mention_encoder, entity_encoder, vocab,
                                        experiment_logdir, finalEvalFlag=0, trainEpoch=-1):
    entire_h1c, entire_h10c, entire_h50c, entire_h64c, entire_h100c, entire_h500c, entire_datapoints = \
        0, 0, 0, 0, 0, 0, 0
    if opts.debug:
        evaluated_world = ['yugioh']
    else:
        evaluated_world = DEV_WORLDS if dev_or_test_flag == 'dev' else TEST_WORLDS

    for world_name in evaluated_world:
        reader_for_eval = WorldsReader(args=opts, world_name=world_name, token_indexers=global_tokenIndexer,
                                       tokenizer=global_tokenizer)
        Evaluator = Evaluate_one_world(args=opts, world_name=world_name,
                                       reader=reader_for_eval,
                                       embedder=textfieldEmbedder,
                                       trainfinished_mention_encoder=mention_encoder,
                                       trainfinished_entity_encoder=entity_encoder,
                                       vocab=vocab, experiment_logdir=experiment_logdir,
                                       dev_or_test=dev_or_test_flag,
                                       berttokenizer=global_tokenizer,
                                       bertindexer=global_tokenIndexer)
        Evaluate_one_world.finalEvalFlag = copy.copy(finalEvalFlag)
        Hits1count, Hits10count, Hits50count, Hits64count, Hits100count, Hits500count, data_points = \
            Evaluator.evaluate_one_world(trainEpoch=trainEpoch)
        entire_h1c += Hits1count
        entire_h10c += Hits10count
        entire_h50c += Hits50count
        entire_h64c += Hits64count
        entire_h100c += Hits100count
        entire_h500c += Hits500count
        entire_datapoints += data_points

    return entire_h1c, entire_h10c, entire_h50c, entire_h64c, entire_h100c, entire_h500c, entire_datapoints

def devEvalExperimentEntireDevWorldLog(experiment_logdir,  t_entire_h1c, t_entire_h10c, t_entire_h50c,
                                       t_entire_h64c, t_entire_h100c, t_entire_h500c, t_entire_datapoints,
                                       epoch=0):
    l = [t_entire_h1c, t_entire_h10c, t_entire_h50c, t_entire_h64c, t_entire_h100c, t_entire_h500c]
    devEvalResultWithPercent = [round(hits_c / t_entire_datapoints * 100, 4) for hits_c in l]
    print('\nt_h1c, h10c, h50c, h64c, h100c, h500c @ Percent:\n', devEvalResultWithPercent)
    dump_dir = experiment_logdir + DevEvalDuringTrainDirForEachExperiment + '/'
    if not os.path.exists(dump_dir):
        os.mkdir(path=dump_dir)
    jpath = dump_dir + 'ep' + str(epoch) + 'devEntireEvalResult.json'
    j = {'t_h1c_Dev': devEvalResultWithPercent[0], 'h10c_Dev':  devEvalResultWithPercent[1],
         'h50c_Dev':  devEvalResultWithPercent[2], 'h64c_Dev':  devEvalResultWithPercent[3],
         'h100c_Dev': devEvalResultWithPercent[4], 'h500c_Dev': devEvalResultWithPercent[5]
         }
    jdump(j=j, path=jpath)