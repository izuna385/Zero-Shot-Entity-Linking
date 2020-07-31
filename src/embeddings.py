from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

class EmbLoader:
    def __init__(self, args):
        self.args = args

    def emb_returner(self):
        if self.args.bert_name == 'bert-base-uncased':
            huggingface_model = 'bert-base-uncased'
        else:
            huggingface_model = 'dummy'
            print(self.args.bert_name,'are not supported')
            exit()
        bert_embedder = PretrainedTransformerEmbedder(model_name=huggingface_model)
        return bert_embedder, bert_embedder.get_output_dim(), BasicTextFieldEmbedder({'tokens': bert_embedder},
                                                                                     allow_unmatched_keys=True)
