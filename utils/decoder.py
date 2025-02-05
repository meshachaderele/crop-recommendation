class FullPipelineWithDecoder:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, X):
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_encoded(self, X):
        y_pred_encoded = self.model.predict(X)
        return y_pred_encoded

    def predict_probability(self, X):
        return self.model.predict_proba(X)