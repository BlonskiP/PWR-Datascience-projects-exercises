from model.utils import load_model

"""
Wrapper around classfier model 
"""
class Classifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def name(self):
        return self.model.name

    def predict(self, x):
        prediction = self.model.predict(x)

        return prediction

