# Configuraciones

Estado: pendiente.

Aquí irán los argumentos de cada corrida, sin tocar código:

- `piloto`: 10 épocas, para validar el pipeline, comparar A contra B y medir el tiempo por época.
- `final`: presupuesto definido en **imágenes procesadas**, no en épocas, según lo medido en el piloto. Lo corre Nicolás en el servidor.

Hardware de referencia (clúster UNABIA): 2x RTX PRO 4000 de 24 GB, 123 GB de RAM, CUDA 12.8, torch 2.11.0+cu128, Python 3.11.16.
