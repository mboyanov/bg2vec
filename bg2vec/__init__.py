from llm2vec import LLM2Vec
class Bg2Vec:

    @staticmethod
    def from_pretrained(model_name_or_path="mboyanov/bg2vec", **kwargs):
        return LLM2Vec.from_pretrained(model_name_or_path, **kwargs)