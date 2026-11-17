#!/usr/bin/env python3
"""Extrait des vidéos d'anime les suites d'images toutes différentes, pour l'entraînement à t quelconque.

L'anime se joue le plus souvent « sur deux » ou « sur trois » : une image tenue ne dit rien du mouvement entre deux
dessins, et l'instant d'une image du milieu n'est connu que si chaque image est un dessin nouveau. On ne garde donc
que les passages animés « sur un » (panoramiques, combats, effets), découpés en suites de --min à --max images :

    sortie/<vidéo>/<image de départ>/frames.npy (N × H × W × 3, uint8) et meta.json

Les images restent brutes : l'entraînement les lit par np.load(mmap_mode="r") et n'en touche que la découpe, sans
rien décoder ; le processeur ne freine plus la carte. Compter 2,8 Mo par image en 720p.

Une suite s'arrête à une image tenue (moins de --visible des pixels changent de plus de 10 niveaux 8 bits : le seul
écart moyen de differs(), dans tools/bench/qualite/sources.py, est trompé par le grain) ou à une coupe (écart moyen
au-dessus de --coupe niveaux). Les intervalles de
--exclure (les passages du banc, mis à part pour juger) ne donnent rien. Seules les vidéos à 23,976 i/s servent :
une autre cadence a souvent été convertie, avec des images mélangées.

    python -I tools/nvof_interp/sequences.py ~/Videos/anime --sortie /ssd/ajp-dataset/suites \\
        --exclure ~/ajp-work/holdout.json --travaux 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

EXTENSIONS = {".mkv", ".mp4", ".m2ts", ".webm"}
CADENCE = Fraction(24000, 1001)


def videos(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTENSIONS)
        elif p.suffix.lower() in EXTENSIONS:
            found.append(p)
    return found


def probe(video: Path) -> tuple[Fraction, int, int, float] | None:
    """Cadence, largeur, hauteur et durée (secondes) de la première piste vidéo."""
    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=avg_frame_rate,r_frame_rate,width,height:format=duration", "-of", "json",
                           str(video)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return None
    info = json.loads(done.stdout)
    s = info["streams"][0]
    avg, real = Fraction(s["avg_frame_rate"]), Fraction(s["r_frame_rate"])
    # Une cadence variable (moyenne différente de la nominale) ne sert pas : l'instant d'une image serait faux.
    return (avg if avg == real else Fraction(0)), int(s["width"]), int(s["height"]), float(info["format"]["duration"])


def excluded(video: Path, holdout: list[dict]) -> list[tuple[float, float]]:
    return [(float(h["debut"]), float(h["fin"])) for h in holdout if Path(h["fichier"]).name == video.name]


def luma(rgb: np.ndarray) -> np.ndarray:
    """Luminance BT.709 sur une grille d'un pixel sur deux, en niveaux 8 bits : assez pour juger une différence."""
    small = rgb[::2, ::2].astype(np.float32)
    return small @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def change(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Écart moyen de luminance et part des pixels qui bougent de plus de 10 niveaux.

    L'écart moyen seul (le critère du banc, 0,7 niveau) voit toutes les images différentes sur une source à grain ou
    à bruit de compression ; une image n'est un dessin nouveau que si une part visible des pixels change."""
    d = np.abs(a - b)
    return float(d.mean()), float((d > 10.0).mean())


def frames(video: Path, width: int, height: int, start: float, seconds: float, hwaccel: bool):
    """Les images de la vidéo de start à start + seconds, en RGB 8 bits, une à une (décodage par ffmpeg)."""
    cmd = ["ffmpeg", "-v", "fatal"] + (["-hwaccel", "cuda"] if hwaccel else []) + [
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{seconds:.3f}", "-map", "0:v:0", "-vf",
        f"scale={width}:{height}:flags=bicubic:in_color_matrix=bt709:in_range=tv:out_range=pc",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = width * height * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def write_run(folder: Path, run: list[np.ndarray], meta: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "frames.npy", np.stack(run))
    # meta.json en dernier : sa présence dit que la suite est complète (precompute_pairs.py s'y fie).
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def extract(video: Path, cap: int, args: argparse.Namespace, holdout: list[dict]) -> str:
    """Les suites d'une vidéo, cherchées dans --fenetres fenêtres de --duree secondes réparties sur toute sa durée :
    tout décoder coûterait un quart d'heure de processeur par épisode, et la diversité vient du nombre de vidéos."""
    info = probe(video)
    if info is None or info[0] != CADENCE:
        return f"{video.name} : ignorée (cadence {info[0] if info else '?'})"
    _, w, h, duration = info
    height = args.hauteur
    width = round(w * height / h / 2) * 2
    skip = excluded(video, holdout)
    out_dir = args.sortie / folder_name(video)
    fps = float(CADENCE)
    kept = frames_kept = 0
    # Fenêtres régulières, en laissant de côté les 2 premières % (logos) ; leur ordre est mêlé pour que le plafond
    # plafond ne garde pas que le début.
    span = max(duration - args.duree, 0.0)
    starts = [span * (0.02 + 0.96 * (i + 0.5) / args.fenetres) for i in range(args.fenetres)]
    starts = starts[::2] + starts[1::2]
    for window in starts:
        if kept >= cap:
            break
        # La fenêtre commence pile sur l'image base : on cherche une demi-image avant, la première image rendue est
        # alors base, et l'indice de chaque image gardée est exact (vérifié par select=eq(n, …)).
        base = round(window * fps)
        if any(window + args.duree > a and window < b for a, b in skip):
            continue
        run: list[np.ndarray] = []
        prev = None
        last_end = -1e9

        def flush(end: int) -> None:
            nonlocal kept, frames_kept, last_end
            # Une suite longue se coupe en morceaux de --max images ; un reste trop court est perdu.
            for first in range(0, len(run) - args.min + 1, args.max):
                piece = run[first : first + args.max]
                index = base + end - len(run) + first
                if len(piece) < args.min or index - last_end < args.espacement * fps or kept >= cap:
                    continue
                meta = {"video": video.name, "image": index, "secondes": round(index / fps, 3), "images": len(piece),
                        "taille": [width, height], "cadence": "24000/1001",
                        "ecart_moyen": round(float(np.mean([change(luma(piece[i + 1]), luma(piece[i]))[0]
                                                            for i in range(len(piece) - 1)])), 3)}
                write_run(out_dir / f"{index:06d}", piece, meta)
                kept += 1
                frames_kept += len(piece)
                last_end = index + len(piece)

        n = 0
        for n, img in enumerate(frames(video, width, height, max(base - 0.5, 0.0) / fps, args.duree, args.hwaccel)):
            y = luma(img)
            new = False
            if prev is not None:
                mean, moving = change(y, prev)
                new = mean <= args.coupe and moving > args.visible
            prev = y
            if new:
                run.append(img)
                continue
            flush(n)
            run = [img]  # image tenue ou coupe : une nouvelle suite commence ici
        flush(n + 1)
    return f"{video.name} : {kept} suites, {frames_kept} images"


def folder_name(video: Path) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in video.stem)[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", type=Path, nargs="*", help="vidéos ou dossiers (parcourus en profondeur)")
    parser.add_argument("--sortie", type=Path, required=True)
    parser.add_argument("--exclure", type=Path, help="JSON [{fichier, debut, fin}] : intervalles mis à part")
    parser.add_argument("--hauteur", type=int, default=720)
    parser.add_argument("--min", type=int, default=5, help="images au moins par suite (écart 4 au moins)")
    parser.add_argument("--max", type=int, default=9, help="images au plus par suite (écart 8 au plus)")
    parser.add_argument("--coupe", type=float, default=30.0, help="écart moyen de luminance d'une coupe, en niveaux")
    parser.add_argument("--visible", type=float, default=0.005,
                        help="part des pixels qui changent de plus de 10 niveaux pour qu'une image soit nouvelle")
    parser.add_argument("--fenetres", type=int, default=12, help="fenêtres lues par vidéo")
    parser.add_argument("--duree", type=float, default=30.0, help="secondes par fenêtre")
    parser.add_argument("--espacement", type=float, default=2.0,
                        help="secondes au moins entre deux suites gardées d'une même fenêtre")
    parser.add_argument("--liste", type=Path, help="fichier : une vidéo par ligne, à la place des sources")
    parser.add_argument("--par-serie", type=int, default=150,
                        help="suites au plus par dossier (une série) : 37 épisodes d'Evangelion ne doivent pas noyer "
                             "un film seul")
    parser.add_argument("--par-video", type=int, default=25, help="suites au plus par vidéo (25 × 9 images ≈ 0,6 Go en 720p)")
    parser.add_argument("--travaux", type=int, default=3, help="vidéos traitées en parallèle")
    parser.add_argument("--hwaccel", action="store_true", help="décodage par le GPU (ffmpeg -hwaccel cuda)")
    args = parser.parse_args()
    if args.min < 3 or args.max < args.min:
        parser.error("il faut 3 <= --min <= --max")

    holdout = json.loads(args.exclure.read_text(encoding="utf-8")) if args.exclure else []
    sources = list(args.sources)
    if args.liste:
        sources += [Path(line.strip()) for line in args.liste.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = videos(sources)
    if not todo:
        parser.error("aucune vidéo")
    # Plafond par série : partagé entre ses épisodes, avant d'écarter ceux déjà extraits.
    per_dir: dict[Path, int] = {}
    for v in todo:
        per_dir[v.parent] = per_dir.get(v.parent, 0) + 1
    caps = {v: min(args.par_video, -(-args.par_serie // per_dir[v.parent])) for v in todo}
    # Une vidéo déjà extraite (son dossier existe) est sautée : on peut relancer après une coupure.
    done_dirs = {p.name for p in args.sortie.iterdir()} if args.sortie.exists() else set()
    todo = [v for v in todo if folder_name(v) not in done_dirs]
    print(f"{len(todo)} vidéos à extraire", flush=True)
    with ProcessPoolExecutor(args.travaux) as pool:
        for line in pool.map(extract, todo, [caps[v] for v in todo], [args] * len(todo), [holdout] * len(todo)):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
