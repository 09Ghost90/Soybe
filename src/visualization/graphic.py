import matplotlib.pyplot as plt
import numpy as np

# Valores obtidos
accuracy = [84.42, 82.35, 78.01, 81.23, 76.02, 78.92, 68.47, 67.86, 75.00]
precision = [84.42, 82.35, 78.01, 81.23, 76.02, 78.92, 68.47, 67.86, 75.00]
recall = [84.42, 82.35, 78.01, 81.23, 76.02, 78.92, 68.47, 67.86, 75.00]
timer_execution = [0,0,0,0,0,0,0,0, 0]
# Verificar adição de um Loss

size_dataset = [5513, 4410, 3308, 2757, 2205, 1654, 1103, 551, 276]

plt.figure(figsize=(10, 6))

plt.plot(size_dataset, accuracy, marker='o', label='Acurácia', linewidth=2)
plt.plot(size_dataset, precision, marker='s', label='Precisão', linewidth=2)
plt.plot(size_dataset, recall, marker='^', label='Revocação', linewidth=2)
plt.plot(size_dataset, timer_execution, marker= "", label='Tempo', linewidth=2)

plt.xlabel('Tamanho do Dataset (nº de imagens)')
plt.ylabel('Percentual (%)')
plt.title('Benchmark - Acurácia, Precisão e Revocação vs. Tamanho do Dataset')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig('benchmark_grafico_linha.png', dpi=300)

