# Projeto Neuro - API de Análise de Dados

## Descrição do Projeto
Este projeto consiste em uma API desenvolvida com **FastAPI** para análise de dados de crédito, incluindo endpoints para:

- **Performance**: cálculo de métricas de desempenho do modelo de crédito, como ROC AUC e volumetria por mês.
- **Aderência**: análise de aderência do modelo usando o teste KS (Kolmogorov-Smirnov).

O projeto também inclui scripts e exemplos para testar a API localmente no **Jupyter Notebook**.

---

## Estrutura do Projeto
ProjetoNeuro/
│
├─ app/ # Código da API
│ └─ main.py # API FastAPI com endpoints de performance e aderência
│
├─ monitoring/ # Arquivos de monitoramento e modelo
│ └─ model.pkl # Modelo pré-treinado
│
├─ datasets/ # Conjuntos de dados (não incluídos no GitHub devido ao tamanho)
│ └─ credit_01/
│
├─ batch_records.json # Exemplo de batch para teste do endpoint de Performance
├─ README.md
└─ .gitignore
ProjetoNeuro/
│
├─ app/ # Código da API
│ └─ main.py # API FastAPI com endpoints de performance e aderência
│
├─ monitoring/ # Arquivos de monitoramento e modelo
│ └─ model.pkl # Modelo pré-treinado
│
├─ datasets/ # Conjuntos de dados (não incluídos no GitHub devido ao tamanho)
│ └─ credit_01/
│
├─ batch_records.json # Exemplo de batch para teste do endpoint de Performance
├─ README.md
└─ .gitignore

Execute a API:
uvicorn app.main:app --reload --port 8001

Teste o endpoint raiz:
GET http://127.0.0.1:8001/

Deve retornar:
{"message": "API Projeto Neuro está rodando!"}


Observações

Arquivos grandes como datasets não foram incluídos no GitHub.

O projeto ainda está em andamento: os endpoints estão funcionando com o modelo model.pkl, mas ajustes e validações adicionais podem ser feitos.

O foco principal foi aprender a integração entre Python, Pandas, FastAPI e Machine Learning.



