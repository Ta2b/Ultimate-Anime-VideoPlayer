#!/usr/bin/env python3
"""Les mires des séances de sonde (chantier 8.1 et 8.12) : deux fichiers à lire en plein écran, une mire par
chapitre.

  patterns.py DOSSIER [--size 2560x1440] [--seconds 6] [--fps 24]
      écrit DOSSIER/patterns-hdr10.mkv et DOSSIER/patterns-sdr.mkv (ffmpeg et libx265 requis)
  patterns.py --selftest

patterns-hdr10.mkv : HEVC 10 bits, PQ BT.2020, plage limitée.
  - fenêtres centrées de 2, 10, 25, 50 et 100 % de l'aire, à 203, 400, 766 et 1 000 nits de contenu, sur noir ;
  - gris 18 % (26 nits, BT.2408) au centre, dans un cadre clair à 1 000 nits qui couvre 0 à 60 % de l'aire
    (l'ABL du panneau : le gris doit rester stable à 3 %).
patterns-sdr.mkv : HEVC 8 bits, BT.709, plage limitée : blanc, gris 25/50/75 %, peau sRGB 255/224/196, en
plein écran.

Lecture : ajp play patterns-hdr10.mkv --fullscreen --mute --no-resume --data-dir=/tmp/ajp-probe, pause à
chaque chapitre, sonde au centre (protocole : docs/reports/sonde-protocole.md).
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

# SMPTE ST 2084.
M1, M2 = 2610 / 16384, 2523 / 4096 * 128
C1, C2, C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_encode(nits):
    y = max(nits, 0.0) / 10000.0
    p = y ** M1
    return ((C1 + C2 * p) / (1 + C3 * p)) ** M2


def pq_decode(code):
    p = max(code, 0.0) ** (1 / M2)
    return 10000.0 * (max(p - C1, 0.0) / (C2 - C3 * p)) ** (1 / M1)


def limited10(e):
    return int(round(64 + 876 * min(max(e, 0.0), 1.0)))


def ycbcr709_limited8(r, g, b):
    """Codes sRGB 8 bits (déjà gamma) vers Y'CbCr BT.709 plage limitée 8 bits."""
    rp, gp, bp = r / 255.0, g / 255.0, b / 255.0
    y = 0.2126 * rp + 0.7152 * gp + 0.0722 * bp
    cb = (bp - y) / 1.8556
    cr = (rp - y) / 1.5748
    return (int(round(16 + 219 * y)), int(round(128 + 224 * cb)), int(round(128 + 224 * cr)))


def window_rect(width, height, area):
    """Rectangle centré, aux proportions de l'image, de la fraction d'aire donnée (en pixels pairs)."""
    side = math.sqrt(area)
    w = max(2, int(round(width * side / 2)) * 2)
    h = max(2, int(round(height * side / 2)) * 2)
    x = (width - w) // 2 // 2 * 2
    y = (height - h) // 2 // 2 * 2
    return x, y, w, h


def hdr_patterns():
    out = []
    for nits in (203, 400, 766, 1000):
        for area in (0.02, 0.10, 0.25, 0.50, 1.00):
            out.append((f"fenêtre {round(area * 100)} % à {nits} nits", "window", area, nits))
    for frame in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        out.append((f"gris 18 % (26 nits), cadre à 1000 nits sur {round(frame * 100)} %", "grey", frame, 26.0))
    return out


def sdr_patterns():
    return [("blanc", (255, 255, 255)), ("gris 25 %", (64, 64, 64)), ("gris 50 %", (128, 128, 128)),
            ("gris 75 %", (191, 191, 191)), ("peau 255/224/196", (255, 224, 196))]


def hdr_frame(width, height, kind, amount, nits):
    """Une image yuv420p10le : planes Y, U, V en petits-boutistes 16 bits."""
    black = limited10(pq_encode(0.0))
    y_plane = bytearray(int.to_bytes(black, 2, "little") * (width * height))
    def fill(x, y, w, h, code):
        value = int.to_bytes(code, 2, "little")
        row = value * w
        for line in range(y, y + h):
            start = (line * width + x) * 2
            y_plane[start:start + 2 * w] = row
    if kind == "window":
        fill(*window_rect(width, height, amount), limited10(pq_encode(nits)))
    else:
        # Le cadre : l'image entière à 1 000 nits moins un trou centré de (1 − amount) de l'aire, puis le gris
        # au centre sur 10 % de l'aire.
        if amount > 0:
            fill(0, 0, width, height, limited10(pq_encode(1000.0)))
            fill(*window_rect(width, height, 1.0 - amount), black)
        fill(*window_rect(width, height, 0.10), limited10(pq_encode(nits)))
    chroma = int.to_bytes(512, 2, "little") * ((width // 2) * (height // 2))
    return bytes(y_plane) + chroma + chroma


def sdr_frame(width, height, rgb):
    y, cb, cr = ycbcr709_limited8(*rgb)
    return bytes([y]) * (width * height) + bytes([cb]) * ((width // 2) * (height // 2)) + \
        bytes([cr]) * ((width // 2) * (height // 2))


def chapters_file(path, titles, seconds):
    with open(path, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for i, title in enumerate(titles):
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={i * seconds * 1000}\nEND={(i + 1) * seconds * 1000}\n")
            f.write(f"title={title}\n")


def encode(frames, titles, out, width, height, seconds, fps, hdr):
    frames_per = int(seconds * fps)
    with tempfile.TemporaryDirectory() as tmp:
        meta = os.path.join(tmp, "chapitres.txt")
        chapters_file(meta, titles, seconds)
        if hdr:
            color = ["-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                     "-colorspace", "bt2020nc", "-color_range", "tv",
                     "-x265-params", "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited:"
                     "hdr10=1:max-cll=1000,400:master-display=G(13250,34500)B(7500,3000)R(34000,16000)"
                     "WP(15635,16450)L(10000000,1)"]
            pix = "yuv420p10le"
        else:
            color = ["-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
                     "-colorspace", "bt709", "-color_range", "tv",
                     "-x265-params", "log-level=error:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"]
            pix = "yuv420p"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-i", meta, "-map", "0:v", "-map_metadata", "1", "-map_chapters", "1",
               "-c:v", "libx265", "-preset", "fast", "-crf", "4", "-g", str(frames_per)] + color + [out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in frames:
            for _ in range(frames_per):
                proc.stdin.write(frame)
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg a échoué pour {out}")


def selftest():
    assert abs(pq_encode(203) - 0.5806) < 1e-3, pq_encode(203)
    assert abs(pq_encode(1000) - 0.7518) < 1e-3, pq_encode(1000)
    for nits in (0.5, 26, 203, 766, 1000):
        assert abs(pq_decode(pq_encode(nits)) - nits) < 1e-6 * max(nits, 1)
    assert limited10(0.0) == 64 and limited10(1.0) == 940
    assert ycbcr709_limited8(255, 255, 255) == (235, 128, 128)
    for area in (0.02, 0.10, 0.25, 0.5, 1.0):
        x, y, w, h = window_rect(2560, 1440, area)
        assert abs(w * h / (2560 * 1440) - area) < 0.01, (area, w, h)
        assert x >= 0 and y >= 0 and x + w <= 2560 and y + h <= 1440
    assert len(hdr_patterns()) == 27
    frame = hdr_frame(64, 36, "window", 0.25, 203)
    assert len(frame) == 64 * 36 * 2 * 3 // 2
    print("patterns : auto-test réussi")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?")
    parser.add_argument("--size", default="2560x1440")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.out:
        parser.error("dossier de sortie attendu")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg introuvable")
    width, height = (int(v) for v in args.size.lower().split("x"))
    if width % 2 or height % 2:
        raise SystemExit("taille paire attendue")
    os.makedirs(args.out, exist_ok=True)
    hdr = hdr_patterns()
    encode((hdr_frame(width, height, kind, amount, nits) for _, kind, amount, nits in hdr), [p[0] for p in hdr],
           os.path.join(args.out, "patterns-hdr10.mkv"), width, height, args.seconds, args.fps, True)
    sdr = sdr_patterns()
    encode((sdr_frame(width, height, rgb) for _, rgb in sdr), [p[0] for p in sdr],
           os.path.join(args.out, "patterns-sdr.mkv"), width, height, args.seconds, args.fps, False)
    print(f"{args.out}/patterns-hdr10.mkv ({len(hdr)} mires), {args.out}/patterns-sdr.mkv ({len(sdr)} mires)")


if __name__ == "__main__":
    sys.exit(main())
