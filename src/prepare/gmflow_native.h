// Le squelette CNN de GMFlow lu pour les noyaux natifs du backend TensorRT (docs/gmflow-natif.md) : les poids
// pris par leurs noms dans la paire ONNX qu'exporte tools/gmfss/export_gmfss.py (initialiseurs
// « gmfss.flownet.backbone.* »), les attributs de ses Conv et de ses InstanceNormalization (nœuds
// « /flownet/backbone/… ») et la normalisation de l'image (/flownet/Sub, /flownet/Div). La structure (blocs,
// raccourcis, trident) est celle du graphe exporté, tenue ici en table : un autre graphe se trahit par un
// nom, une forme ou un attribut qui manque ou diffère.
#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "core/types.h"
#include "prepare/gmflow_backbone.h"
#include "prepare/gmflow_upsampler.h"

namespace kirari::prepare {

// Le squelette et son plan : la 7 × 7 de pas 2, les six blocs résiduels (64, 96, 128 canaux), la 1 × 1 et le
// trident aux poids partagés (pas 1 vers le quart, pas 2 vers le huitième). La paire peut être en FP16 ou en
// FP32. Refus : ModelUnreadable (ONNX illisible), FieldMissing (un nom, une forme ou un attribut que les
// noyaux ne font pas, avec son nom).
std::expected<GmflowBackbone, Refusal> gmflow_backbone(
    std::span<const std::byte> pair, GmflowStemInput input = GmflowStemInput::kFolded);

// La 3 × 3 (flot xy puis 128 caractéristiques, Relu) et la 1 × 1 (9 masques de 4 × 4) de la paire.
std::expected<GmflowUpsampler, Refusal> gmflow_upsampler(std::span<const std::byte> pair);

// Le plan se tient-il pour les noyaux ? Chaque opération lit des tenseurs déjà écrits et des statistiques
// déjà faites sur ce tenseur, écrit un tenseur neuf de la taille que son pas demande, et ne demande que ce
// que les noyaux font : la 7 × 7 sur l'image seule, des canaux d'entrée par 32, de sortie par 16 (128 au
// plus), la normalisation à la lecture des seules 3 × 3 de pas 1. Refus FieldMissing.
std::expected<void, Refusal> gmflow_backbone_check(const GmflowBackbone& backbone);

}  // namespace kirari::prepare
