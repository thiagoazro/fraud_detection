# Fraud Detection — MLOps (Disciplina Anhanguera)

Pipeline completo de detecção de fraudes em transações financeiras, cobrindo todas as etapas de MLOps: treinamento com rastreamento de experimentos via MLflow, serving com FastAPI, containerização com Docker e orquestração com Kubernetes.

---

## Visão Geral

O projeto treina um modelo **XGBoost** em um dataset sintético de 1 milhão de transações (`transactions_1M.csv`) para classificar cada transação como fraudulenta ou legítima. O modelo treinado é registrado no **MLflow Model Registry** com o alias `champion` e disponibilizado via uma **API REST** que retorna a probabilidade de fraude para lotes de transações.

### Features utilizadas

| Feature | Descrição |
|---|---|
| `amount` | Valor da transação (R$) |
| `hour` | Hora do dia (0–23) |
| `dow` | Dia da semana (0=segunda, 6=domingo) |
| `channel` | Canal (0=presencial, 1=online) |
| `international` | Transação internacional (0/1) |
| `new_merchant` | Comerciante novo (0/1) |
| `acct_age_days` | Idade da conta em dias |
| `txn_count_24h` | Número de transações nas últimas 24h |
| `txn_amount_24h` | Valor total transacionado nas últimas 24h (R$) |
| `distance_km` | Distância da última transação (km) |
| `device_change` | Mudança de dispositivo (0/1) |
| `ip_risk` | Score de risco do IP (0.0–1.0) |

---

## Estrutura do Projeto

```
Fraud Detection/
├── data/
│   └── transactions_1M.csv        # Dataset de transações
├── src/
│   ├── train.py                   # Script de treinamento + registro no MLflow
│   └── serve.py                   # API FastAPI para inferência
├── docker/
│   ├── Dockerfile.mlflow          # Imagem do servidor MLflow
│   ├── Dockerfile.train           # Imagem do job de treinamento
│   └── Dockerfile.serve           # Imagem da API de serving
├── k8s/
│   ├── 00-namespace.yaml          # Namespace do projeto
│   ├── 01-mlflow-pvc.yaml         # PersistentVolumeClaim para o MLflow
│   ├── 02-mlflow-deployment.yaml  # Deployment do servidor MLflow
│   ├── 03-mlflow-service.yaml     # Service do MLflow
│   ├── 04-train-job.yaml          # Job de treinamento
│   ├── 05-serve-deployment.yaml   # Deployment da API
│   ├── 06-serve-service.yaml      # Service da API
│   ├── 07-data-pvc.yaml           # PVC para os dados
│   └── 08-upload-pod.yaml         # Pod auxiliar para upload de dados
├── notebook_mlops.ipynb            # Notebook principal do projeto MLOps
├── notebook_precision_recall.ipynb # Análise de precision/recall e threshold
├── notebook_docker_kubernetes.ipynb# Experimentos com Docker e Kubernetes
├── api-test.py                    # Script de teste da API
├── requirements-train.txt         # Dependências do treinamento
└── requirements-serve.txt         # Dependências do serving
```

---

## Como Executar

### Com Kubernetes (produção)

Aplique os manifests em ordem:

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-mlflow-pvc.yaml
kubectl apply -f k8s/07-data-pvc.yaml
kubectl apply -f k8s/02-mlflow-deployment.yaml
kubectl apply -f k8s/03-mlflow-service.yaml
kubectl apply -f k8s/08-upload-pod.yaml   # faz upload do CSV
kubectl apply -f k8s/04-train-job.yaml    # executa o treinamento
kubectl apply -f k8s/05-serve-deployment.yaml
kubectl apply -f k8s/06-serve-service.yaml
```

### Localmente (desenvolvimento)

**Treinamento:**
```bash
pip install -r requirements-train.txt
MLFLOW_TRACKING_URI=http://localhost:5000 DATA_PATH=data/transactions_1M.csv python src/train.py
```

**Serving:**
```bash
pip install -r requirements-serve.txt
MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

---

## API de Inferência

### `GET /health`
Retorna o status da API e confirma se o modelo está carregado.

```json
{"status": "ok", "model_uri": "models:/fraud-xgb@champion", "model_loaded": true}
```

### `POST /predict`
Recebe um lote de transações e retorna a probabilidade de fraude para cada uma.

**Request:**
```json
{
  "x": [
    [50.0, 10, 1, 0, 0, 0, 800, 1, 50.0, 0.5, 0, 0.05],
    [3500.0, 3, 6, 1, 1, 1, 30, 8, 4200.0, 2800.0, 1, 0.92]
  ]
}
```

**Response:**
```json
{"predictions": [0.0312, 0.9847]}
```

Probabilidade > 0.5 indica transação **suspeita**.

### Testando a API

Com o port-forward ativo:
```bash
kubectl port-forward svc/fraud-serve-svc 8000:8000 -n fraud-detection
python api-test.py
```

---

## Modelo e Treinamento

- **Algoritmo:** XGBoost (`binary:logistic`)
- **Métrica de otimização:** AUCPR (Area Under Precision-Recall Curve)
- **Desbalanceamento:** tratado via `scale_pos_weight = negativos / positivos`
- **Threshold:** selecionado automaticamente para atingir precisão alvo (`TARGET_PRECISION=0.90` por padrão), maximizando recall
- **Rastreamento:** todos os hiperparâmetros, métricas e artefatos (matriz de confusão, metadata) são logados no MLflow
- **Registro:** o modelo é registrado no MLflow Model Registry com alias `champion`

### Hiperparâmetros configuráveis (via variáveis de ambiente)

| Variável | Padrão | Descrição |
|---|---|---|
| `MAX_DEPTH` | 6 | Profundidade máxima das árvores |
| `ETA` | 0.08 | Learning rate |
| `SUBSAMPLE` | 0.9 | Fração de amostras por árvore |
| `COLSAMPLE` | 0.9 | Fração de features por árvore |
| `NUM_BOOST_ROUND` | 300 | Número de árvores |
| `TARGET_PRECISION` | 0.90 | Precisão alvo para seleção do threshold |
| `SAMPLE_N` | 0 | Amostrar N linhas (0 = usar tudo) |

---

## Dependências Principais

| Biblioteca | Uso |
|---|---|
| `xgboost 2.1.1` | Treinamento do modelo |
| `mlflow 2.14.3` | Rastreamento de experimentos e model registry |
| `scikit-learn 1.5.2` | Métricas e split de dados |
| `fastapi 0.112.0` | API de serving |
| `uvicorn 0.30.6` | Servidor ASGI |
| `pydantic 2.7.4` | Validação de requests |

---

## Contexto Acadêmico

Projeto desenvolvido para a disciplina de **MLOps** da **Anhanguera**, demonstrando na prática:
- Rastreamento de experimentos com MLflow
- Containerização de pipelines de ML com Docker
- Orquestração com Kubernetes
- Serving de modelos via API REST
- Boas práticas de engenharia de ML (separação treino/serving, versionamento de modelos, threshold calibrado)
