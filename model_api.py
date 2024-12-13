import json
import torch

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class DistilBertApi:
    def __init__(self, trained_model_dir='./model', label_mapping_file='./model/label_mapping.json'):

        self.model = DistilBertForSequenceClassification.from_pretrained(trained_model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(trained_model_dir)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"device in use: {self.device}")
        self.model = self.model.to(self.device)

        with open(label_mapping_file, 'r') as f:
            self.label_mapping = json.load(f)

    def predict_top_5(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1).squeeze().tolist()

        top_5_indices = torch.topk(torch.tensor(probabilities), 5).indices.tolist()
        top_5_results = {self.label_mapping[str(idx)]: probabilities[idx] for idx in top_5_indices}

        sorted_results = sorted(top_5_results.items(), key=lambda item: item[1], reverse=True)

        formatted_results = [(label, f"{int(round(prob * 100, 0))}%") for label, prob in sorted_results]

        formatted_results = [
            (label, f"{int(round(prob * 100, 0))}%")
            for label, prob in sorted_results
            if prob >= 0.01
        ]

        return formatted_results
