# Audit 25. Assistant de premier lancement

**Audit par zone. Priorité : haute.** 6 fichiers, 1 443 lignes ; tests rattachés : 2 fichiers, 317 lignes.

## Mission

L'assistant : ordre des écrans, choix des moteurs, écran Prêt, calibration, ce qui est enregistré et quand.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Choix écrits avant la calibration, reprise après une fermeture.
- Durée annoncée honnête, TensorRT compris (UX-05).
- « Passer » annule bien ses choix (UX-06).
- Mise à jour proposée seulement là où elle est possible (UX-07).
- Assistant relancé pendant une lecture (UX-02).
- Écran Moteurs sauté sans explication (UX-14).

## Frontières

Dépend de :

- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (6 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (6 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (6 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (4 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (4 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (4 inclusions)
- [20. Socle commun (core)](20-socle-core.md) (4 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (2 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (1 inclusion)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (1 inclusion)
- [43. Textes et traductions](43-textes-traductions.md) (1 inclusion)
- [12. Cadence et horloges](12-cadence-horloges.md) (1 inclusion)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (3 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (3 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (2 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (2 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)
- [47. Export vidéo](47-export.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (1 inclusion)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| UX-05 | Majeur | La durée de « Optimise now » est calculée avant le choix de TensorRT. Le Ready annonce environ 5 min au lieu d'environ 20, téléchargement non compris. |
| UX-06 | Majeur | « Make it the default video player » s'applique même après « Skip ». Le résumé n'en dit rien. |
| UX-07 | Majeur | `updates_managed` n'est jamais rempli. L'assistant propose « Automatic » sur macOS ou dans Program Files, où rien ne s'installera. |
| TRT-18 | Mineur | Dossiers temporaires à nom fixe, partagés entre utilisateurs puis effacés par `remove_all`. |
| UX-09 | Mineur | Textes et unités français en dur dans l'interface anglaise (« Go », « i/s », « actif »). Le test i18n ne voit que `tr()`. |
| UX-14 | Mineur | Quand TensorRT est impossible (pilote, disque), l'écran Moteurs est sauté sans explication. |
| UX-23 | Mineur | Dans l'assistant, « q » quitte d'un coup, et ni Entrée ni Tab ne marchent. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/engine_screen.cpp` | 416 |
| `src/app/play/engine_screen.h` | 72 |
| `src/app/play/first_run.cpp` | 580 |
| `src/app/play/first_run.h` | 141 |
| `src/app/play/wizard.cpp` | 149 |
| `src/app/play/wizard.h` | 85 |

## Tests rattachés

- `tests/unit/app/first_run_test.cpp`
- `tests/unit/app/wizard_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
