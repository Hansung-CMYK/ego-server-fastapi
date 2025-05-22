import torch
import numpy as np
from transformers import Pipeline

class MultiLabelPipeline(Pipeline):
    def __init__(self, model, tokenizer, threshold=0.3, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.threshold = threshold
        self.id2label = model.config.id2label

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        return self.tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")

    def _forward(self, model_inputs):
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.sigmoid(logits)
        return {"probs": probs}

    def postprocess(self, model_outputs):
        probs = model_outputs["probs"].detach().cpu().numpy()
        results = []
        for sample in probs:
            labels = []
            scores = []
            for i, score in enumerate(sample):
                if score >= self.threshold:
                    label = self.model.config.id2label.get(i, str(i))
                    labels.append(label)
                    scores.append(float(score))
            results.append({"labels": labels, "scores": scores})
        return results

