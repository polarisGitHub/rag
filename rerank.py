import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from domain.information import RerankerModelInfo


class Reranker(object):

    def __init__(self, model: RerankerModelInfo) -> None:
        self.__model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model.model_path)
        self.model.eval()

    def rerank(self, text: str, data: list[str], text_mapper) -> list[dict]:
        pairs = []
        rerank_results = []
        pairs = [[text, item] for item in list(map(text_mapper, data))]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = (self.model(**inputs, return_dict=True).logits.view(-1).float()).tolist()
            index = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            for i in index:
                rerank_results.append(
                    {
                        "data": data[i],
                        "score": scores[i],
                    }
                )
        return rerank_results
