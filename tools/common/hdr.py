"""Constantes et lecteurs que les outils de mesure partagent (B46-18) : une seule copie du PQ (ST 2084), des
matrices BT.709 → BT.2020 → LMS → ICtCp et du lecteur PPM 16 bits.

Un outil l'importe en ajoutant ce dossier à son chemin, même sous `python -I` :

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
    import hdr
"""
import numpy as np

# SMPTE ST 2084 (PQ), en fractions exactes.
PQ_M1 = 2610 / 16384
PQ_M2 = 2523 / 4096 * 128
PQ_C1 = 3424 / 4096
PQ_C2 = 2413 / 4096 * 32
PQ_C3 = 2392 / 4096 * 32
PQ_PEAK_NITS = 10000.0

# Rec. 2020 vers LMS puis ICtCp (ITU-R BT.2100) ; BT.709 vers BT.2020 (lumière linéaire).
BT709_TO_2020 = np.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]])
BT2020_TO_LMS = np.array([[1688, 2146, 262], [683, 2951, 462], [99, 309, 3688]]) / 4096.0
LMS_TO_ICTCP = np.array([[2048, 2048, 0], [6610, -13613, 7003], [17933, -17390, -543]]) / 4096.0
LUMA_2020 = np.array([0.2627, 0.6780, 0.0593])
LUMA_709 = np.array([0.2126, 0.7152, 0.0722])


def pq_to_linear(code):
    """Codes PQ (0 à 1) vers la lumière divisée par 10 000 nits."""
    p = np.power(np.clip(code, 0.0, 1.0), 1.0 / PQ_M2)
    return np.power(np.maximum(p - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * p), 1.0 / PQ_M1)


def linear_to_pq(linear):
    """Lumière divisée par 10 000 nits vers les codes PQ (0 à 1)."""
    q = np.power(np.clip(linear, 0.0, 1.0), PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * q) / (1.0 + PQ_C3 * q), PQ_M2)


def nits_to_pq(nits):
    """Inverse de l'EOTF PQ : nits vers code [0, 1]."""
    return linear_to_pq(np.asarray(nits, dtype=np.float64) / PQ_PEAK_NITS)


def pq_to_nits(code):
    return pq_to_linear(np.asarray(code, dtype=np.float64)) * PQ_PEAK_NITS


def read_ppm16(path, strict16=True):
    """Un PPM binaire (P6), poids fort d'abord, en flottants [0, 1] de forme (hauteur, largeur, 3).

    strict16 : exige 65535 comme maximum (les instantanés PQ du lecteur) ; sinon tout maximum est mis à l'échelle."""
    with open(path, "rb") as f:
        data = f.read()
    fields = []
    i = 0
    while len(fields) < 4:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] not in (b"\n", b""):
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        fields.append(data[i:j])
        i = j
    i += 1
    if fields[0] != b"P6":
        raise ValueError(f"{path} : PPM binaire (P6) attendu")
    width, height, maxval = int(fields[1]), int(fields[2]), int(fields[3])
    if strict16 and maxval != 65535:
        raise ValueError(f"{path} : PPM 16 bits (P6, 65535) attendu")
    dtype = ">u2" if maxval > 255 else "u1"
    pixels = np.frombuffer(data, dtype=dtype, count=width * height * 3, offset=i)
    return pixels.reshape(height, width, 3).astype(np.float64) / maxval
