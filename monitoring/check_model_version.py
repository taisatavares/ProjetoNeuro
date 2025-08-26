import pickle
import sklearn

# Caminho para o model.pkl
model_path = "C:/Users/taisa/PycharmProjects/ProjetoNeuro/monitoring/model.pkl"


with open(model_path, "rb") as f:
    model = pickle.load(f)

print("Modelo carregado com sucesso!")
print("Versão atual do scikit-learn no seu ambiente:", sklearn.__version__)
