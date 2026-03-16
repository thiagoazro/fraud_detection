# Execute este bloco com o port-forward ativo
import requests

BASE_URL = "http://localhost:8000"

# Teste de health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Teste de predição
# Features na ordem: amount, hour, dow, channel, international,
#                    new_merchant, acct_age_days, txn_count_24h,
#                    txn_amount_24h, distance_km, device_change, ip_risk
transacoes = {
    "x": [
        # Transação normal: valor baixo, horário comercial, conta antiga
        [50.0,  10, 1, 0, 0, 0, 800, 1,  50.0,  0.5, 0, 0.05],
        # Transação suspeita: valor alto, madrugada, internacional, dispositivo novo, IP de risco
        [3500.0, 3, 6, 1, 1, 1,  30, 8, 4200.0, 2800.0, 1, 0.92],
    ]
}

response = requests.post(f"{BASE_URL}/predict", json=transacoes)
result = response.json()

print("\nPredições (probabilidade de fraude):")
for i, prob in enumerate(result["predictions"]):
    tipo = "SUSPEITA" if prob > 0.5 else "normal"
    print(f"  Transação {i+1}: {prob:.4f} ({tipo})")