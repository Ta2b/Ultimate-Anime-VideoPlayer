# Carte des documents

Commencer par [`CLAUDE.md`](../CLAUDE.md) à la racine : le guide de reprise (ce qu'est le lecteur, où est quoi,
comment construire et tester, les pièges). Ensuite, selon le sujet :

## À jour (décrivent le code actuel)

| Document | Sujet |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Modules, threads, graphe des cibles, règles de dépendance |
| [`adr/`](adr/README.md) | Décisions d'architecture, une page chacune |
| [`moteurs.md`](moteurs.md) | Moteurs des réseaux : Vulkan, TensorRT, Core ML ; MIGraphX et OpenVINO retirés ; livraison de TensorRT |
| [`interpolation.md`](interpolation.md) | Moteurs du mouvement : RIFE, DistilDRBA ; GMFSS retiré |
| [`auto-libre.md`](auto-libre.md) | Mode Auto de l'interpolation : Rapide et Stable (le § 3, mesures d'octobre 2026, est historique) |
| [`hdr-anime.md`](hdr-anime.md) | Le grade HDR anime guidé ; la sortie HDR ou SDR qui suit le système |
| [`luts.md`](luts.md) | Les LUT 3D du grade (le choix des looks n'est plus dans le menu) |
| [`ux-parametres.md`](ux-parametres.md) | Le menu des paramètres : section 0 pour l'arbre en vigueur, le reste est l'historique du tri |
| [`nvidia-linux.md`](nvidia-linux.md), [`nvidia-windows.md`](nvidia-windows.md) | TensorRT et NVIDIA par système |
| [`amd-windows.md`](amd-windows.md) | AMD et Windows : le profil RDNA4 du moteur Vulkan, D3D11VA |
| [`span-fp8-cutlass.md`](span-fp8-cutlass.md) | SPAN entier en FP8 (cran Performance, RTX 40) |
| [`fp8-interpolation.md`](fp8-interpolation.md) | Le FP8 de RIFE et DistilDRBA ; celui de GMFSS retiré |
| [`rife-fusionne.md`](rife-fusionne.md), [`drba-fusionne.md`](drba-fusionne.md) | Réseaux coupés en moteurs TensorRT et noyaux CUDA |
| [`rife-candidats.md`](rife-candidats.md) | RIFE 4.25 lite et 4.26 face à 4.22 lite et 4.25 : mesures sur processeur, niveaux choisis, protocole ABAB sur la RTX |
| [`signature-des-releases.md`](signature-des-releases.md) | Signer le manifeste des mises à jour ; signature des exécutables Windows et macOS |
| [`anime4k.md`](anime4k.md) | Anime4K, l'agrandissement en shaders des petites cartes ; le banc des petites cartes |
| [`banc-qualite.md`](banc-qualite.md) | Banc de qualité des réseaux contre leur référence FP32 et la vraie image, sans carte graphique |
| [`parite-windows-linux.md`](parite-windows-linux.md) | Banc de parité Windows contre Linux sur le même PC |
| [`test-utilisateurs.md`](test-utilisateurs.md) | Test avec cinq utilisateurs : tâches et grille |
| [`a-faire-1-0.md`](a-faire-1-0.md) | Ce qui reste avant la 1.0 |
| [`phase6-a-tester.md`](phase6-a-tester.md) | Pistes d'optimisation codées, à mesurer sur une vraie carte |

## Historiques (gardés comme trace, marqués en tête)

Plans et suivis de version (`plan-v0.5.md`, `plan-v0.6.md`, `tests-v0.5-windows-macos.md`), passation et état
d'un poste (`handoff-pc.md`, `etat-windows.md`, `tests-pc.md`), audits (`audit-fonctions.md`, `audit-noms.md`,
`audit-ui.md`) et relevés datés ([`reports/`](reports/)). Ils parlent de fonctions retirées depuis (profils,
HDR adaptatif et sa calibration, RTX HDR et TrueHDR, méthode HDR à choisir, ancien arbre du menu) : ne pas les
prendre pour l'état du code.

## Retiré, pour ne pas le chercher

| Retiré | Quand | Ce qui le remplace |
|---|---|---|
| RTX HDR, Maxine, émulation TrueHDR | ADR 0001, PR #530 | le grade HDR anime seul |
| HDR adaptatif et sa calibration | PR #530 | le grade HDR anime (lumière fixe en HDR, marge des éclats par plan en SDR) |
| Profils (Ctrl+1 à 9, profil de démarrage) | PR #531 | `play.toml` garde la dernière chaîne ; `user.toml` ne garde que `[keys]` |
| Méthode HDR (bouton, écran de l'assistant, `hdr.engine`) | PR #532 | `system_hdr` (`src/app/play/host_facts.h`) |
| Ancien arbre en cinq catégories | PR #533 | Vidéo, Audio et lecture, Général ([ADR 0012](adr/0012-lecteur-simple.md)) |
| MIGraphX, OpenVINO, Windows ML | 0.10.0 | Vulkan sur AMD et Intel |
| GMFSS Fortuna (base, union, fusionné, FP8) | 0.10.0 | RIFE et DistilDRBA |
