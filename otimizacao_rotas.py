import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(42)
horas = np.arange(0, 24, 1)
tempo_viagem = np.sin(horas / 24 * 2 * np.pi) + np.random.normal(0, 0.1, len(horas))

scaler = MinMaxScaler()
tempo_viagem_scaled = scaler.fit_transform(tempo_viagem.reshape(-1, 1))

df = pd.DataFrame({"hora": horas, "tempo_viagem": tempo_viagem_scaled.flatten()})

def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 3
X, y = create_sequences(tempo_viagem_scaled, seq_length)

X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=8, verbose=1)

predictions = model.predict(X)
predicted_traffic = scaler.inverse_transform(predictions)

G = nx.Graph()
cities = ["A", "B", "C", "D", "E"]
for city in cities:
    G.add_node(city)

edges = [
    ("A", "B", 10 + predicted_traffic[0][0]), 
    ("B", "C", 15 + predicted_traffic[1][0]),
    ("C", "D", 5 + predicted_traffic[2][0]), 
    ("D", "E", 8 + predicted_traffic[3][0]), 
    ("A", "D", 20 + predicted_traffic[4][0])
]

G.add_weighted_edges_from(edges)

source, target = "A", "E"
shortest_path = nx.shortest_path(G, source=source, target=target, weight='weight')
print(f"Melhor rota de {source} para {target}: {shortest_path}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de previsão de tráfego
axes[0].plot(horas[seq_length:], scaler.inverse_transform(y.reshape(-1,1)), label="Real")
axes[0].plot(horas[seq_length:], predicted_traffic, label="Previsto", linestyle="dashed")
axes[0].set_xlabel("Hora do Dia")
axes[0].set_ylabel("Tempo de Viagem (Min)")
axes[0].legend()
axes[0].set_title("Previsão de Tráfego")

pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, ax=axes[1])
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.1f} min" for k, v in labels.items()}, ax=axes[1])
axes[1].set_title("Rede de Rotas com Previsão de Tráfego")

plt.tight_layout()
plt.show()
