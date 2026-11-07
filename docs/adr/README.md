# Décisions d'architecture

Une page par décision : le contexte, ce qui a été décidé, ce que cela entraîne. Une décision remplacée garde
sa page, avec le numéro de celle qui la remplace.

| N° | Décision | État |
|---|---|---|
| [0001](0001-retrait-rtx-hdr.md) | Retrait de RTX HDR et de Maxine : le grade du lecteur seul | Acceptée (#264) |
| [0002](0002-span-une-fois-par-source.md) | SPAN une fois par image source, le warp après SPAN | Acceptée (#268, #286, #287) |
| [0003](0003-un-seul-mode-auto.md) | Un seul mode Auto : Rapide par défaut, Stable au choix | Acceptée (#287) |
| [0004](0004-grade-scene-donnee-par-le-lecteur.md) | Le grade en trois passes, la scène mesurée par le lecteur | Acceptée (#283, #284) |
| [0007](0007-span-basse-precision-par-defaut.md) | SPAN basse précision par défaut, derrière une porte durcie et un reçu v2 | Acceptée (#365 à #376), mesures RTX à faire |
| [0008](0008-anneau-rgb10a2.md) | Anneau RGB10A2 derrière un interrupteur ; TensorRT n'écrit pas dans l'anneau | Acceptée (#378 à #383), défaut à mesurer |
| [0009](0009-grade-dans-l-anneau.md) | Le grade exact dans l'anneau ; la LUT réservée au chantier 8 | En cours (chantier 2c) |
| [0010](0010-rapide-a-pas-constant-par-plan.md) | Rapide à pas constant par plan, derrière `--fast-steady` | En cours (chantier 7, #400 à #403), mesures à faire |
| [0011](0011-presentation-en-boucle-fermee.md) | Présentation en boucle fermée : un correctif de libplacebo, pas une swapchain maison | En cours (chantier 6) |
| [0012](0012-lecteur-simple.md) | Un lecteur simple : un seul HDR qui suit le système, pas de profils, menu en trois catégories | Acceptée (#530 à #533) |
| [0013](0013-ed25519-maison-garde.md) | La vérification Ed25519 maison est gardée, sous vecteurs RFC 8032 et Wycheproof | Acceptée |
