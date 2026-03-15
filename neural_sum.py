# neural_sum.py
import random

# Parámetros
N_HIDDEN = 1000
LEARNING_RATE = 0.000126
ITERATIONS = 20000
ESCALA = 100.0

# Entradas
x1 = float(input("Primer número: "))
x2 = float(input("Segundo número: "))
resultado_correcto = x1 + x2

# Normalización
x1_norm = x1 / ESCALA
x2_norm = x2 / ESCALA
resultado_norm = resultado_correcto / ESCALA

# Inicialización de pesos y bias
hidden_weights = [[random.random()*0.1, random.random()*0.1] for _ in range(N_HIDDEN)]
hidden_biases = [random.random()*0.1 for _ in range(N_HIDDEN)]
output_weights = [random.random()*0.1 for _ in range(N_HIDDEN)]
output_bias = random.random()*0.1

print("\nComenzando entrenamiento...")

for iteration in range(ITERATIONS):
    # Forward pass
    hidden_outputs = []
    for i in range(N_HIDDEN):
        h = hidden_weights[i][0]*x1_norm + hidden_weights[i][1]*x2_norm + hidden_biases[i]
        hidden_outputs.append(h)
    
    y_pred = sum([output_weights[i]*hidden_outputs[i] for i in range(N_HIDDEN)]) + output_bias
    
    # Error
    error = resultado_norm - y_pred
    
    # Ajuste de pesos de salida
    for i in range(N_HIDDEN):
        output_weights[i] += LEARNING_RATE * error * hidden_outputs[i]
    output_bias += LEARNING_RATE * error
    
    # Ajuste de pesos de la capa oculta
    for i in range(N_HIDDEN):
        # Aplicamos un factor de activación lineal limitada
        grad = error * output_weights[i]
        hidden_weights[i][0] += LEARNING_RATE * grad * x1_norm
        hidden_weights[i][1] += LEARNING_RATE * grad * x2_norm
        hidden_biases[i] += LEARNING_RATE * grad
    
    # Mostrar progreso cada 100 iteraciones
    if iteration % 100 == 0:
        print(f"Iteración: {iteration}, Predicción: {y_pred*ESCALA:.6f}, Error: {error*ESCALA:.6f}")

print("\nEntrenamiento terminado")
print("Resultado final:", y_pred*ESCALA)
