# Pipeline Derma

Repositorio asociado a la tesis **“Impacto del balanceo sintético con StyleGAN2 en el rendimiento de un clasificador DINOv2 para lesiones cutáneas del dataset BCN20000”**, desarrollada por **Jeobardo Jerif Cornejo Cuno** en la Universidad Nacional del Altiplano de Puno (2026).

## Resumen

Esta investigación evalúa el uso de imágenes dermatoscópicas sintéticas como estrategia para reducir el desbalance de clases en la clasificación automatizada de lesiones cutáneas. El estudio combina un modelo generativo **StyleGAN2 condicional** con un clasificador **DINOv2 multihead**, capaz de resolver simultáneamente una tarea multiclase de 11 categorías diagnósticas y una tarea binaria de lesiones benignas y malignas.

El experimento se desarrolló sobre el dataset **BCN20000**. Se comparó una línea base entrenada únicamente con imágenes reales frente a ocho escenarios con datos sintéticos, construidos con proporciones de 25 %, 50 %, 75 % y 100 % mediante dos esquemas de balanceo: `deficit_fill` y `total_mix`. Cada escenario se ejecutó diez veces con semillas controladas; las imágenes sintéticas se incorporaron exclusivamente al conjunto de entrenamiento, mientras que la validación y la prueba conservaron únicamente imágenes reales.

## Objetivo de la investigación

Evaluar el balanceo de clases con StyleGAN2 como estrategia para mitigar el sesgo, midiendo su efecto en la sensibilidad y el rendimiento de un clasificador DINOv2 multihead para la detección de lesiones malignas minoritarias en BCN20000.

## Pipeline experimental

1. **Preparación de datos:** depuración de metadatos, organización de las 11 clases diagnósticas y construcción de las divisiones de entrenamiento, validación y prueba.
2. **Generación sintética:** entrenamiento de StyleGAN2 condicional con imágenes de 256 × 256 píxeles para producir nuevas muestras por clase.
3. **Evaluación y filtrado:** medición de la calidad generativa mediante FID, KID y Precision–Recall for Distributions. Las muestras sintéticas se filtran por clase utilizando embeddings de Inception v3, PCA, subcentroides KMeans y distancia coseno.
4. **Construcción de escenarios:** combinación controlada de imágenes reales y sintéticas mediante los esquemas `deficit_fill` y `total_mix`.
5. **Clasificación multihead:** uso de `dinov2_vitb14_reg` como extractor de características congelado, con una cabeza binaria y otra multiclase.
6. **Evaluación estadística:** comparación de los escenarios mediante Balanced Accuracy, AUC-ROC, F1-score, precisión y recall, junto con pruebas t pareadas, Cochran’s Q, McNemar con corrección de Bonferroni e intervalos bootstrap al 95 %.

## Escenarios evaluados

| Esquema | Proporciones | Descripción |
| --- | --- | --- |
| `base_line` | Sin síntesis | Conserva la distribución original de las imágenes reales. |
| `deficit_fill` | 25 %, 50 %, 75 % y 100 % | Mantiene todas las imágenes reales y añade muestras sintéticas para cubrir parcial o totalmente el déficit de las clases minoritarias. |
| `total_mix` | 25 %, 50 %, 75 % y 100 % | Define un tamaño común por clase y modifica la composición de datos reales y sintéticos según la proporción seleccionada. |

## Resultados principales

- StyleGAN2 obtuvo un **FID global de 7.02** y un **KID de 0.0021**, aunque la fidelidad fue heterogénea entre clases y disminuyó en varias categorías con pocas muestras reales.
- La **Balanced Accuracy multiclase** pasó de **0.424** en la línea base a **0.298** en `total_mix/100`.
- El AUC-ROC se mantuvo alrededor de **0.83** en los escenarios moderados, pero descendió en `total_mix/100`.
- El mayor recall binario se observó en `deficit_fill/50`, con **0.836 ± 0.037**, frente a **0.828 ± 0.035** en la línea base; esta mejora no estuvo acompañada por una mejora multiclase.
- `deficit_fill` produjo una degradación menor que `total_mix` al comparar proporciones equivalentes.
- Ninguno de los escenarios con imágenes sintéticas superó el rendimiento multiclase de la línea base. Las diferencias de Balanced Accuracy frente al escenario base fueron estadísticamente significativas.

## Conclusión

Bajo las condiciones evaluadas, el balanceo sintético con StyleGAN2 no mejoró el rendimiento general del clasificador DINOv2. Un buen resultado global de fidelidad generativa no garantiza que las imágenes sintéticas aporten diversidad clínica ni características discriminativas útiles para el clasificador. El trabajo aporta evidencia sobre los límites del balanceo sintético y muestra la importancia de evaluar por separado la calidad generativa, la sensibilidad por clase, la calibración y el rendimiento de cada tarea.

## Tecnologías principales

- Python 3.10
- PyTorch y torchvision
- StyleGAN2 condicional
- DINOv2 (`dinov2_vitb14_reg`)
- scikit-learn y SciPy
- Jupyter Notebook
- CUDA para aceleración por GPU

## Alcance

Este repositorio documenta un trabajo de investigación en dermatología computacional. Los modelos y resultados presentados tienen fines académicos y experimentales; no constituyen una herramienta de diagnóstico clínico ni sustituyen la evaluación de un profesional de la salud.
