# TransNetV2, pour comparer

`tnet.py` réécrit TransNetV2 (Souček et Lokoč, 2020, licence MIT) ; seuls ses poids viennent de l'extérieur :
`transnetv2-pytorch-weights.pth` du paquet PyPI `transnetv2-pytorch` 1.0.5 (`pip download --no-deps transnetv2-pytorch`,
puis extraire le .whl), chargé avec `torch.load(..., weights_only=True)`.

    python3 -I eval.py transnetv2-pytorch-weights.pth vidéo...

Écrit, par vidéo, le numéro de la première image de chaque plan (probabilité de coupe > 0,5) et `<vidéo>.npy`
(la probabilité de chaque image). Il faut torch et ffmpeg.
