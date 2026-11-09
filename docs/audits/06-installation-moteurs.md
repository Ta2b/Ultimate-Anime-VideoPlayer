# Audit 06. Installation des moteurs et téléchargements

**Audit par zone. Priorité : très haute.** 14 fichiers, 2 082 lignes ; tests rattachés : 4 fichiers, 1 068 lignes.

## Mission

Le téléchargement et l'installation de TensorRT et des modèles : index, reprise par plage, proxy, décompression, empreintes, liens durs, place disque.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Reprise : `.part` nommé par contenu, versions mélangées (TRT-10, SEC-14).
- Proxy Windows : PAC, WPAD, exclusions, authentification (WIN-05).
- Place disque : estimation avec liens durs (TRT-11).
- Module périmé ou d'une autre ABI : pas de boucle de réinstallation (TRT-03).
- Décompression : plafond avant le contrôle d'empreinte.
- Annulation, taille et débit affichés (UX-12).

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (12 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (3 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (1 inclusion)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)
- [12. Cadence et horloges](12-cadence-horloges.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (1 inclusion)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (1 inclusion)

Utilisé par :

- [25. Assistant de premier lancement](25-assistant.md) (6 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (4 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (3 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (3 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (3 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (3 inclusions)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (3 inclusions)
- [20. Socle commun (core)](20-socle-core.md) (2 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (1 inclusion)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (1 inclusion)
- [28. CI, publication et signature](28-ci-publication.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| SEC-14 | Mineur | La décompression des fichiers NVIDIA n'a pas de plafond avant le contrôle du SHA. Le `.part` est repris par son nom (voir TRT-10). |
| TRT-03 | Mineur | Sur `AbiMismatch`, le module est retéléchargé et réinstallé à chaque lancement, sans jamais réussir. Démarrage lent et environ 1 Go rehaché. |
| TRT-10 | Mineur | Le `.part` d'un fichier NVIDIA porte son nom, pas son SHA. Deux versions peuvent être recollées, ce qui n'est vu qu'après 650 Mo (voir SEC-14). |
| TRT-11 | Mineur | L'estimation de place ignore la reprise par liens durs. Une mise à jour est refusée « pas assez de place » à tort. |
| TRT-17 | Mineur | Défauts groupés : échec de `nvidia-smi` gardé, carte inconnue en « ptx », moteurs reconstruits à chaque pilote sans purge, doc MPEG-4 fausse, largeur en 16/9 sur ultra-large. |
| UX-09 | Mineur | Textes et unités français en dur dans l'interface anglaise (« Go », « i/s », « actif »). Le test i18n ne voit que `tr()`. |
| UX-12 | Mineur | Le téléchargement de 650 Mo n'a ni annulation, ni taille, ni débit. |
| WIN-05 | Mineur | Le proxy Windows est lu à moitié : ni PAC, ni WPAD, ni exclusions, ni authentification (voir SEC-13). |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/engine_downloads.cpp` | 173 |
| `src/app/play/engine_downloads.h` | 32 |
| `src/app/play/engine_install.cpp` | 608 |
| `src/app/play/engine_install.h` | 128 |
| `src/app/play/engine_tools.cpp` | 121 |
| `src/app/play/engine_tools.h` | 36 |
| `src/core/engines.cpp` | 60 |
| `src/core/engines.h` | 93 |
| `src/core/inflate.cpp` | 58 |
| `src/core/inflate.h` | 25 |
| `src/core/model_index.cpp` | 454 |
| `src/core/model_index.h` | 155 |
| `src/core/sha256_blocks.cpp` | 125 |
| `src/core/sha256_blocks.h` | 14 |

## Tests rattachés

- `tests/unit/app/engine_install_test.cpp`
- `tests/unit/core/engines_test.cpp`
- `tests/unit/core/inflate_test.cpp`
- `tests/unit/core/model_index_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
