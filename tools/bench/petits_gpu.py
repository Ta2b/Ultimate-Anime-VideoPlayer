#!/usr/bin/env python3
"""Banc des petites cartes (docs/anime4k.md) : ce qu'Économe peut agrandir sur un iGPU ou une carte sans unités
matricielles. Il décide kAnime4kInAuto et kLightSpanInAuto (src/app/play/auto_config.h).

Deux mesures, sur le poste à mesurer (binaire livré, pas linux-prefix en Debug) :

- Temps : chaque variante joue le même extrait en plein écran, sans interpolation (--rife=0), deux fois en
  alternance (A, B, A, B…) ; le relevé --stats donne les images perdues et le temps GPU de l'affichage
  (Anime4K y tourne), le profil GPU (.gpu.tsv) celui des moteurs (SPAN, ArtCNN).
- Qualité : chaque variante exporte le même plan 1080p réduit à 540 lignes par le lecteur, puis agrandi ×2 ;
  PSNR et SSIM contre le plan d'origine (ffmpeg). Économe (sans agrandissement) sert de repère : sa sortie
  540 lignes est agrandie par ffmpeg (Lanczos), comme l'écran le ferait.

  python3 -I tools/bench/petits_gpu.py --video EPISODE.mkv --sortie DOSSIER \\
      [--debut 300] [--secondes 30] [--tours 2] [--ajp ajp] \\
      [--span FICHIER.onnx] [--artcnn FICHIER.onnx] [--modele NOM=FICHIER.onnx]... [--option=--software-gpu]
  python3 -I tools/bench/petits_gpu.py --selftest

Sortie : DOSSIER/resume.md (le tableau à recopier dans la PR ou docs/anime4k.md), DOSSIER/resume.json, et une
image du milieu du plan par variante (DOSSIER/qualite/*.png) pour juger à l'œil.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

STATS_COLUMNS = 14  # tools/bench/banc_resume.py : COLUMNS
SPAN_PERFORMANCE = "2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
ARTCNN = "2x_ArtCNN_C4F32_rgb_fp16.onnx"


def variants(args):
    """(nom, options du lecteur) : Économe d'abord, le repère ; les autres agrandissent depuis --hauteur."""
    out = [("econome", ["--no-upscale"]),
           ("anime4k-fast", ["--model=anime4k-fast"]),
           ("anime4k-hq", ["--model=anime4k-hq"])]
    if args.span:
        out.append(("span-performance", [f"--model={args.span}"]))
    if args.artcnn:
        out.append(("artcnn", [f"--model={args.artcnn}"]))
    for extra in args.modele:
        name, _, path = extra.partition("=")
        if not path:
            sys.exit(f"--modele attend NOM=FICHIER : {extra}")
        out.append((name, [f"--model={path}"]))
    return out


def run(command, log=None):
    with open(log, "w", encoding="utf-8") if log else open(os.devnull, "w", encoding="utf-8") as sink:
        done = subprocess.run(command, stdout=sink, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              check=False)
    return done.returncode


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def read_stats(path):
    """Images vidéo montrées, perdues (compteurs cumulés dropped et missing de la dernière ligne, comme
    banc_resume.py), et le temps GPU de l'affichage (µs) des images neuves."""
    shown = 0
    last = None
    display = []
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        at = {name: i for i, name in enumerate(header)}
        if len(header) < STATS_COLUMNS or "display_gpu_us" not in at:
            raise ValueError(f"{path} : en-tête inattendu")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            last = fields
            if fields[at["fresh"]] != "1":
                continue
            shown += 1
            if int(fields[at["display_gpu_us"]] or 0) > 0:
                display.append(int(fields[at["display_gpu_us"]]))
    lost = int(last[at["dropped"]]) + int(last[at["missing"]]) if last else 0
    return shown, lost, display


def read_engines(path):
    """La somme des médianes des étapes des moteurs (ms par passe), du profil GPU ; 0 sans moteur."""
    if not os.path.exists(path):
        return 0.0
    total = 0.0
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 4:
                total += float(fields[3])
    return total


ANIME4K_LINE = re.compile(r"profil GPU · Anime4K : médiane ([0-9.]+) ms, p95 ([0-9.]+) ms")


def read_anime4k(log):
    """Médiane et p95 du temps GPU des passes d'Anime4K (ms), lus au journal ; None sans elles."""
    if not os.path.exists(log):
        return None
    with open(log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            found = ANIME4K_LINE.search(line)
            if found:
                return float(found.group(1)), float(found.group(2))
    return None


def timing(args, plays):
    rows = {}
    for name, options in plays:
        # Économe joue la source telle quelle ; les autres travaillent à la hauteur d'Économe.
        height = "--work-height=0" if name == "econome" else f"--work-height={args.hauteur}"
        runs = []
        for round_ in range(1, args.tours + 1):
            base = os.path.join(args.sortie, "temps", f"{name}-{round_}")
            command = [args.ajp, "play", args.video, f"--start={args.debut}", f"--seconds={args.secondes}",
                       "--fullscreen", "--mute", "--no-resume", "--rife=0",
                       f"--data-dir={os.path.join(args.sortie, 'donnees')}", f"--stats={base}.tsv", height,
                       *options, *args.option]
            code = run(command, f"{base}.log")
            if code != 0 or not os.path.exists(f"{base}.tsv"):
                print(f"échec : {name}, tour {round_} (voir {base}.log)", file=sys.stderr)
                continue
            shown, dropped, display = read_stats(f"{base}.tsv")
            runs.append({"images": shown, "perdues": dropped,
                         "affichage_ms": percentile(display, 50) / 1000.0,
                         "affichage_p95_ms": percentile(display, 95) / 1000.0,
                         "moteurs_ms": read_engines(f"{base}.gpu.tsv"),
                         "anime4k": read_anime4k(f"{base}.log")})
        rows[name] = runs
    return rows


def ffmpeg_scores(ffmpeg, candidate, reference, upscale=False):
    """PSNR moyen (dB) et SSIM de candidate contre reference ; upscale : candidate agrandie d'abord (Lanczos)."""
    graph = "[0:v]scale=1920:1080:flags=lanczos[c];[c][1:v]" if upscale else "[0:v][1:v]"
    scores = {}
    for metric, pattern in (("psnr", r"average:([0-9.inf]+)"), ("ssim", r"All:([0-9.]+)")):
        done = subprocess.run([ffmpeg, "-hide_banner", "-nostats", "-i", candidate, "-i", reference,
                               "-lavfi", graph + metric, "-f", "null", "-"],
                              capture_output=True, text=True, check=False)
        found = re.findall(pattern, done.stderr)
        scores[metric] = float(found[-1]) if found else None
    return scores


def quality(args, plays):
    folder = os.path.join(args.sortie, "qualite")
    os.makedirs(folder, exist_ok=True)
    reference = os.path.join(folder, "reference.mkv")
    # Le plan d'origine en 1080p, sans perte : la vérité.
    run([args.ffmpeg, "-y", "-ss", str(args.debut), "-t", str(args.plan), "-i", args.video, "-an", "-sn",
         "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "ffv1", "-pix_fmt", "yuv420p10le", reference])
    rows = {}
    for name, options in plays:
        out = os.path.join(folder, f"{name}.mkv")
        if os.path.exists(out):
            os.remove(out)
        # Le lecteur réduit à 540 lignes (le travail d'Économe sur un écran 1080p), agrandit ×2 (sauf Économe),
        # exporte.
        command = [args.ajp, "play", reference, "--mute", "--no-resume", "--rife=0", "--window=960x540",
                   f"--data-dir={os.path.join(args.sortie, 'donnees')}", f"--keys=0.5:export-lossless={out}",
                   "--seconds=1", "--work-height=540", *options, *args.option]
        run(command, os.path.join(folder, f"{name}.log"))
        if not os.path.exists(out):
            print(f"échec de l'export : {name} (voir {folder}/{name}.log)", file=sys.stderr)
            continue
        rows[name] = ffmpeg_scores(args.ffmpeg, out, reference, upscale=name == "econome")
        run([args.ffmpeg, "-y", "-ss", str(args.plan / 2), "-i", out, "-frames:v", "1",
             os.path.join(folder, f"{name}.png")])
    return rows


def table(time_rows, quality_rows):
    lines = ["| Variante | Images perdues | Affichage (ms, médiane / p95) | Moteurs (ms) | Anime4K (ms) "
             "| PSNR (dB) | SSIM |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in dict.fromkeys([*time_rows, *quality_rows]):
        runs = time_rows.get(name, [])
        shown = sum(r["images"] for r in runs)
        dropped = sum(r["perdues"] for r in runs)
        total = shown + dropped
        lost = f"{dropped} sur {total} ({100.0 * dropped / total:.1f} %)" if total else "—"
        display = (f"{statistics.median(r['affichage_ms'] for r in runs):.2f} / "
                   f"{max(r['affichage_p95_ms'] for r in runs):.2f}") if runs else "—"
        engines = f"{statistics.median(r['moteurs_ms'] for r in runs):.2f}" if runs else "—"
        a4k = [r["anime4k"] for r in runs if r["anime4k"]]
        a4k_text = f"{statistics.median(a[0] for a in a4k):.2f}" if a4k else "—"
        q = quality_rows.get(name, {})
        psnr = f"{q['psnr']:.2f}" if q.get("psnr") is not None else "—"
        ssim = f"{q['ssim']:.4f}" if q.get("ssim") is not None else "—"
        lines.append(f"| {name} | {lost} | {display} | {engines} | {a4k_text} | {psnr} | {ssim} |")
    return "\n".join(lines) + "\n"


def selftest():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert ANIME4K_LINE.search("[app] profil GPU · Anime4K : médiane 3.21 ms, p95 4.50 ms (900 images, …)")
    rows = {"econome": [{"images": 100, "perdues": 0, "affichage_ms": 1.0, "affichage_p95_ms": 2.0,
                         "moteurs_ms": 0.0, "anime4k": None}],
            "anime4k-fast": [{"images": 100, "perdues": 3, "affichage_ms": 4.0, "affichage_p95_ms": 6.0,
                              "moteurs_ms": 0.0, "anime4k": (3.0, 5.0)}]}
    text = table(rows, {"anime4k-fast": {"psnr": 38.5, "ssim": 0.98}})
    assert "| anime4k-fast | 3 sur 103 (2.9 %) | 4.00 / 6.00 | 0.00 | 3.00 | 38.50 | 0.9800 |" in text, text
    assert "| econome | 0 sur 100 (0.0 %)" in text, text
    print("selftest : ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video")
    parser.add_argument("--sortie")
    parser.add_argument("--debut", type=float, default=300.0, help="début de l'extrait, en secondes")
    parser.add_argument("--secondes", type=int, default=30, help="durée de chaque lecture chronométrée")
    parser.add_argument("--plan", type=float, default=4.0, help="durée du plan exporté pour la qualité")
    parser.add_argument("--tours", type=int, default=2)
    parser.add_argument("--hauteur", type=int, default=540,
                        help="hauteur de travail chronométrée (Économe : la moitié de l'écran, 540 en 1080p)")
    parser.add_argument("--ajp", default="ajp")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--span", help=f"SPAN Performance ({SPAN_PERFORMANCE})")
    parser.add_argument("--artcnn", help=f"ArtCNN ({ARTCNN})")
    parser.add_argument("--modele", action="append", default=[], help="autre modèle, NOM=FICHIER.onnx")
    parser.add_argument("--option", action="append", default=[],
                        help="option de plus pour chaque lancement du lecteur (--software-gpu pour un essai)")
    parser.add_argument("--sans-temps", action="store_true")
    parser.add_argument("--sans-qualite", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.video or not args.sortie:
        parser.error("il faut --video et --sortie")
    os.makedirs(os.path.join(args.sortie, "temps"), exist_ok=True)
    plays = variants(args)
    time_rows = {} if args.sans_temps else timing(args, plays)
    quality_rows = {} if args.sans_qualite else quality(args, plays)
    text = table(time_rows, quality_rows)
    with open(os.path.join(args.sortie, "resume.md"), "w", encoding="utf-8") as out:
        out.write(text)
    with open(os.path.join(args.sortie, "resume.json"), "w", encoding="utf-8") as out:
        json.dump({"temps": time_rows, "qualite": quality_rows}, out, indent=2, ensure_ascii=False)
    print(text)


if __name__ == "__main__":
    main()
