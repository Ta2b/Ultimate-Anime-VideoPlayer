# Graphes SPAN quantifiés FP8 (E4M3, Q/DQ) du banc

Tous : entrée `input` 1×3×H×W FP16, sortie `output`, opset 19, ~1 Mo. Poids en constantes FP8, échelle par tenseur ; Q/DQ devant chaque Conv.
- `span_balanced_fp8_pertensor_all.onnx` : SPAN Balanced HD V3.1 (b8f64), motif de base (make_fp8_qdq.py --weights per-tensor --acts all). Mesure TensorRT 11.3, 4070 Ti, 1088×1920 : 18,3 à 19,0 ms contre 15,6 à 15,9 ms en FP16.
- `span_balanced_fp8_qout_att.onnx` : le même avec Q/DQ ajouté sur la sortie de la conv 1×1 d'attention et de c3 : 18,1 à 18,2 ms.
- `span_noatt_fp8_pertensor.onnx` / `span_noatt_fp16.onnx` : SPAN sans l'attention (Tanh/Mul remplacés par l'identité, mesure de vitesse seulement, pas un modèle utilisable) : FP16 12,65 à 13,38 ms ; FP8 13,54 à 13,60 ms. Motif qui fusionne le mieux (chaîne résiduelle 64 canaux 544×960) : Conv → Q → DQ → Add(saut) → Q : −17 % contre FP16.
Calibration : maxima sur 3 images synthétiques (pas de calibration d'anime réelle) : pour étudier la structure du graphe, pas la qualité.
