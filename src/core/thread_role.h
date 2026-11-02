// La place d'un thread face à la lecture : qui passe devant quand le processeur est plein.
//
// Les threads de la lecture (rendu, calcul GPU, audio) doivent tenir leur cadence même quand un travail de
// fond (la pré-analyse des plans, qui décode tout le fichier) occupe tous les cœurs. Chaque thread déclare
// son rôle au début de son corps ; le système en fait ce qu'il sait faire :
//   - Windows : MMCSS (« Playback », « Audio ») pour la lecture, priorité basse pour le fond ;
//   - Linux : nice 10 pour le fond (hérité par les threads qu'il crée ensuite, ceux de FFmpeg compris) ;
//     la lecture garde la priorité normale (la relever demande des droits) ;
//   - macOS : classes de qualité de service (interactive, utilitaire).
// Jamais d'échec : un système qui refuse laisse le thread tel qu'il est.
#pragma once

namespace ajp {

enum class ThreadRole {
  Playback,    // rendu, calcul GPU de la lecture
  Audio,       // sortie audio
  Background,  // analyse, téléchargement : ne doit jamais retarder la lecture
};

// Applique le rôle au thread appelant, pour toute sa vie.
void set_thread_role(ThreadRole role) noexcept;

}  // namespace ajp
