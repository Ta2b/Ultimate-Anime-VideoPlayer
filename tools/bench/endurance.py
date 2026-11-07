#!/usr/bin/env python3
"""Banc d'endurance (P2.44) : un seul lecteur joue des épisodes à la suite pendant des heures, et la mémoire,
la VRAM, les poignées et le journal sont relevés chaque minute. Linux et Windows, Python seul (aucun paquet).

  python3 tools/bench/endurance.py --video=EPISODE.mkv --sortie=DOSSIER [--heures=8] [--ajp=CHEMIN]
                                   [--intervalle=60] [-- OPTIONS DU LECTEUR]

Le dossier de sortie reçoit 400 liens vers la même vidéo (ep001.mkv, ep002.mkv...) : la lecture enchaîne
l'épisode suivant (réglage autoplay_next dans un play.toml propre au banc), donc la chaîne se rouvre à chaque
épisode, comme une soirée de série. Le lecteur est arrêté au bout des heures demandées.

Sortie : releves.tsv (une ligne par relevé), rapport.md (pentes après la première heure et verdict), et le
dossier de réglages du banc avec ajp.log. Le verdict échoue quand une pente dépasse son plafond : une fuite
d'une poignée par épisode ou de quelques Mo par heure se voit sur 8 h, pas sur une minute.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Plafonds des pentes, par heure, mesurés à partir de la deuxième heure (préchauffage, caches et moteurs
# chargés). Une lecture saine reste plate à quelques Mo près.
PLAFONDS = {"rss_mio": 32.0, "vram_mio": 64.0, "poignees": 20.0}
JOURNAL_MAX_MIO = 3 * 32 + 1  # ajp.log et ses deux parties tournées (P2.43)
LIENS = 400


def lier(video: Path, dossier: Path) -> None:
    dossier.mkdir(parents=True, exist_ok=True)
    for i in range(1, LIENS + 1):
        lien = dossier / f"ep{i:03d}{video.suffix}"
        if lien.exists():
            continue
        try:
            os.link(video, lien)  # lien dur : sans droits sous Windows, sur le même volume
        except OSError:
            os.symlink(video, lien)


def memoire_et_poignees(pid: int):
    """La mémoire résidente (Mio) et le nombre de poignées (fichiers ouverts sous Linux)."""
    if sys.platform.startswith("linux"):
        rss = 0.0
        with open(f"/proc/{pid}/status", encoding="utf-8") as status:
            for ligne in status:
                if ligne.startswith("VmRSS:"):
                    rss = int(ligne.split()[1]) / 1024
        return rss, len(os.listdir(f"/proc/{pid}/fd"))
    if sys.platform == "win32":
        sortie = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"$p = Get-Process -Id {pid}; \"$($p.WorkingSet64) $($p.HandleCount)\""],
            capture_output=True, text=True, check=False).stdout.split()
        if len(sortie) == 2:
            return int(sortie[0]) / (1024 * 1024), int(sortie[1])
    return float("nan"), -1


def vram_mio() -> float:
    """La VRAM occupée sur le GPU (nvidia-smi), le banc tournant seul ; NaN sans pilote NVIDIA."""
    if not shutil.which("nvidia-smi"):
        return float("nan")
    sortie = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False).stdout.split()
    return float(sortie[0]) if sortie else float("nan")


def journal_mio(reglages: Path) -> float:
    return sum(f.stat().st_size for f in reglages.glob("ajp*.log*")) / (1024 * 1024)


def pente(points):
    """Moindres carrés : unité par heure ; None à moins de trois points."""
    points = [(t, v) for t, v in points if v == v and v >= 0]
    if len(points) < 3:
        return None
    n = len(points)
    mt = sum(t for t, _ in points) / n
    mv = sum(v for _, v in points) / n
    num = sum((t - mt) * (v - mv) for t, v in points)
    den = sum((t - mt) ** 2 for t, _ in points)
    return num / den * 3600 if den else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--sortie", required=True, type=Path)
    parser.add_argument("--heures", type=float, default=8.0)
    parser.add_argument("--intervalle", type=float, default=60.0)
    parser.add_argument("--ajp", default="ajp")
    parser.add_argument("options", nargs="*")
    args = parser.parse_args()

    sortie = args.sortie.resolve()
    episodes = sortie / "episodes"
    reglages = sortie / "reglages"
    lier(args.video.resolve(), episodes)
    reglages.mkdir(parents=True, exist_ok=True)
    (reglages / "play.toml").write_text("[playlist]\nautoplay_next = true\n", encoding="utf-8")

    commande = [args.ajp, "play", str(episodes / f"ep001{args.video.suffix}"), "--fullscreen", "--no-resume",
                f"--data-dir={reglages}", *args.options]
    debut = time.monotonic()
    fin = debut + args.heures * 3600
    releves = []
    with open(sortie / "lecteur.txt", "w", encoding="utf-8") as console, \
            open(sortie / "releves.tsv", "w", encoding="utf-8") as tsv:
        tsv.write("secondes\trss_mio\tvram_mio\tpoignees\tjournal_mio\n")
        lecteur = subprocess.Popen(commande, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT)
        while time.monotonic() < fin and lecteur.poll() is None:
            time.sleep(args.intervalle)
            if lecteur.poll() is not None:
                break
            try:
                rss, poignees = memoire_et_poignees(lecteur.pid)
            except OSError:
                break
            ligne = (time.monotonic() - debut, rss, vram_mio(), poignees, journal_mio(reglages))
            releves.append(ligne)
            tsv.write("\t".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in ligne) + "\n")
            tsv.flush()
        code = lecteur.poll()
        if code is None:
            lecteur.terminate()
            try:
                lecteur.wait(30)
            except subprocess.TimeoutExpired:
                lecteur.kill()

    duree = (releves[-1][0] if releves else 0) / 3600
    stables = [r for r in releves if r[0] >= 3600]
    lignes = [f"# Banc d'endurance : {duree:.2f} h", "",
              f"commande : `{' '.join(commande)}`", ""]
    echec = []
    if code is not None:
        echec.append(f"le lecteur s'est arrêté seul (code {code}) au bout de {duree:.2f} h")
    if duree < args.heures * 0.98:
        echec.append(f"durée courte : {duree:.2f} h sur {args.heures} h")
    lignes.append("| relevé | début | fin | pente par heure (après 1 h) | plafond |")
    lignes.append("|---|---|---|---|---|")
    for colonne, nom in ((1, "rss_mio"), (2, "vram_mio"), (3, "poignees")):
        p = pente([(r[0], r[colonne]) for r in stables])
        valeurs = [r[colonne] for r in releves]
        texte = "—" if p is None else f"{p:+.1f}"
        premier, dernier = (f"{valeurs[0]:.1f}", f"{valeurs[-1]:.1f}") if valeurs else ("—", "—")
        lignes.append(f"| {nom} | {premier} | {dernier} | {texte} | {PLAFONDS[nom]} |")
        if p is not None and p > PLAFONDS[nom]:
            echec.append(f"{nom} monte de {p:.1f} par heure (plafond {PLAFONDS[nom]})")
    taille = max((r[4] for r in releves), default=0.0)
    lignes.append(f"| journal_mio | | {taille:.1f} | | {JOURNAL_MAX_MIO} au plus |")
    if taille > JOURNAL_MAX_MIO:
        echec.append(f"le journal pèse {taille:.1f} Mio")
    episodes_vus = journal_episodes(reglages)
    lignes += ["", f"épisodes enchaînés (journal) : {episodes_vus}", "",
               "verdict : " + ("réussi" if not echec else "échec"), *[f"- {e}" for e in echec]]
    (sortie / "rapport.md").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print("\n".join(lignes))
    return 0 if not echec else 1


def journal_episodes(reglages: Path) -> int:
    total = 0
    for fichier in reglages.glob("ajp*.log*"):
        with open(fichier, encoding="utf-8", errors="replace") as journal:
            total += sum("épisode suivant" in ligne for ligne in journal)
    return total


if __name__ == "__main__":
    sys.exit(main())
