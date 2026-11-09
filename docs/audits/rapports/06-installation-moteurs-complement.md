# Complément de l'audit 06. Installation des moteurs et téléchargements : les renvois reçus

Branche : `claude/project-thread-af4o84`. Commit audité : `f8cb2cd`. Date : `2026-11-09`.

## Verdict

Les trois renvois reçus tiennent. Le premier requalifie A06-02 : le scénario « Jérôme » est faux sous Windows 11
(le manifeste UTF-8 et la page 1252 font arriver le « é » intact), et seul reste le cas d'un nom hors de la page
ANSI du système, dans `curl.exe` et `tar.exe` (doublon d'A20-02, gravité ramenée à Mineur). Le deuxième est un
doublon d'A06-03 et d'A26-01, avec un ordre de correction qui règle aussi A06-09. Le troisième ouvre un défaut
neuf, mineur : A06-13, `curl.exe` sans `--ssl-revoke-best-effort`, qui casse tout téléchargement derrière un
antivirus qui inspecte le TLS.

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les renvois de `audits-docs/complements/06.md` dans le code,
conclure chacun par confirmé, réfuté ou doublon, et signaler toute baisse de qualité d'image faite pour la vitesse
sans mesure (aucune dans cette zone : elle ne touche que le transport et l'installation). Aucune question posée à
Tatoub. Le conteneur n'a ni Windows ni GPU : tout est lu sur `f8cb2cd`, où les numéros de ligne des renvois n'ont
pas bougé.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A06-13 | Mineur | `src/app/play/engine_tools.cpp:76-87`, `src/app/play/updater.cpp:88-91` | Sous Windows, `curl.exe` de System32 passe par Schannel, qui vérifie la révocation du certificat et **refuse** la connexion quand il ne peut pas la vérifier (`CRYPT_E_NO_REVOCATION_CHECK`, `CRYPT_E_REVOCATION_OFFLINE`, code 35 de curl). Ni `curl_download_args` (installation des moteurs) ni `fetch` (mise à jour, qui n'utilise pas `curl_download_args` et ne reprend donc pas non plus le proxy de Windows) ne passent `--ssl-revoke-best-effort` (curl 7.70 et plus, présent dans tout Windows 11). `--retry-all-errors` fait en plus réessayer cinq fois une erreur qui ne passera jamais. | Un antivirus grand public qui inspecte le HTTPS (Avast, AVG, Kaspersky, ESET avec « filtrage SSL ») présente un certificat signé par sa propre racine, sans point de distribution de liste de révocation ni OCSP : chaque connexion échoue. De même sur un réseau qui bloque les serveurs OCSP de l'autorité (Wi-Fi d'hôtel, portail captif, pare-feu strict). TensorRT ne s'installe jamais (« échec à l'étape téléchargement », Vulkan à la place) et la recherche de mise à jour échoue, alors que le navigateur, qui lui tolère un serveur de révocation injoignable, charge GitHub sans souci. | Sous `_WIN32`, ajouter `--ssl-revoke-best-effort` dans `curl_download_args` et dans `fetch` (le mieux : que `fetch` parte de `curl_download_args`, ce qui lui donne aussi le proxy). Le risque est faible : un certificat **effectivement** révoqué reste refusé quand le serveur de révocation répond, seul un serveur injoignable est toléré, et chaque fichier est de toute façon vérifié par son SHA-256 (`core::verify_file`, `engine_install.cpp:449`, 475 ; SHA de chaque fichier de NVIDIA, l. 385). Reconnaître le code 35 dans `curl_failure` pour afficher « certificate check failed (antivirus or proxy inspecting HTTPS?) » au lieu d'un échec muet. | Lu : `grep -rn revoke src tools` ne trouve aucune option de révocation ; `updater.cpp:88-91` construit ses propres arguments. À confirmer : Windows 11 avec Avast et l'« Agent Web » actif, cocher TensorRT ; le journal montre un code 35 de curl. |

## Renvois traités

| # | De | Où | Conclusion | Preuve et note |
|---|---|---|---|---|
| 1 | 20 | `src/app/play/engine_install.cpp:293` | Confirmé : **A06-02 requalifié** (Majeur vers Mineur), risque restant doublon d'**A20-02** | Chaque exécutable MSVC du projet embarque `activeCodePage = UTF-8` (`cmake/utf8.manifest:5`, `cmake/WindowsUtf8.cmake`, appelé à la fin de `CMakeLists.txt:113-114`). Dans `ajp.exe` sous Windows 11, `GetACP()` vaut donc 65001 et `chunk.string()`, `part.string()`, `staging.string()` (l. 293, 462, 493) rendent de l'UTF-8 sans exception ; `widen` (`process_win.cpp:60-67`, l. 128) les relit en `CP_UTF8` et `CreateProcessW` transmet une ligne de commande UTF-16 exacte. Le « é » de « Jérôme » arrive donc intact jusqu'à l'enfant ; si celui-ci lit `main` dans la page ANSI, Windows le convertit en 1252, où « é » existe. **Le scénario d'A06-02 est faux**, comme l'exception dans le thread d'installation. Reste le seul cas d'A20-02 : un nom hors de la page ANSI du système (profil « Тарас » sur un Windows français) arrive en « ? » à `curl.exe` ou `tar.exe` s'ils lisent `main` et non `wmain`, ce que rien ne dit pour les binaires de System32. La correction d'A06-02 (`core::to_utf8`) reste bonne pour la portabilité et pour baisser `kMax` de `tools/ci/check-path-string.sh`, mais ne suffit pas pour ce cas : il faut celle d'A20-02 (dossier de travail ASCII, ou un test qui prouve que les deux outils lisent `wmain`). À confirmer : Windows 11 français, compte local « Тарас », cocher TensorRT. |
| 2 | 26 | `src/app/play/engine_downloads.cpp:130-166` | Doublon d'**A06-03** et d'**A26-01** (correction commune), confirmé | Les deux défauts sont dans la même fonction, `update_stale_engines` : l. 130 ne réinstalle que sur `AbiMismatch` (A26-01 : un SHA changé à ABI égale n'arrive jamais, correctifs de noyaux et de qualité compris), et l. 134-140 ne compare que la licence de `paquet.txt`, jamais son SHA (A06-03 : même SHA, réinstallation à chaque lancement). `paquet.txt` porte bien `sha256=` (`engine_install.cpp:541`). Ordre à tenir pour une correction unique : **lire `paquet.txt` avant d'ouvrir le module** (l. 125-128). SHA différent de l'index : réinstaller sans charger le module, ce qui évite aussi qu'une DLL TensorRT chargée bloque le renommage sous Windows (A06-09). SHA égal et `AbiMismatch` : ne rien retélécharger, prévenir et passer à Vulkan. Note absente (installation antérieure à la note) : réinstaller une fois. Un seul test couvre les trois cas. |
| 3 | 34 | `src/app/play/engine_tools.cpp:76-87` | Confirmé : **A06-13** (neuf) | Voir la table des défauts. Le renvoi visait `curl_download_args` ; le même manque est dans `updater.cpp:88-91`, qui n'appelle pas `curl_download_args`. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A06-02 | Requalifié | Mineur, et seulement pour un nom hors de la page ANSI du système, dans les enfants (renvoi 1). |
| WIN-05 | Toujours présent, étendu | La mise à jour (`updater.cpp:88-91`) ne lit même pas le proxy manuel que lit l'installation. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 26 | `src/app/play/updater.cpp:88-91` | `fetch` construit ses propres arguments de curl : ni proxy de Windows (WIN-05), ni `--ssl-revoke-best-effort` (A06-13). À aligner sur `curl_download_args`. |

## Points positifs

- Le lecteur lui-même est en UTF-8 sous Windows 11 (manifeste), et `run_process` passe par `CreateProcessW` avec une
  ligne de commande UTF-16 : la conversion est juste jusqu'à la frontière de l'enfant.
- Chaque fichier téléchargé est vérifié par son SHA-256 : une vérification de révocation tolérante n'ouvre pas la
  porte à un fichier altéré.

## Questions ouvertes

Aucune.
