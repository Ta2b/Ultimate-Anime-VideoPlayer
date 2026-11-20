#include "prepare/gmflow_propagation.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>

#include "prepare/gmflow_attention.h"
#include "prepare/gmflow_onnx_view.h"

namespace kirari::prepare {
namespace {

namespace onnx = gpu::ir::onnx;
constexpr int kD = kGmflowModel;
constexpr std::string_view kPrefix = "/flownet/feature_flow_attn/";

struct Projection {
  GmflowProjection data;
  std::string input;
  std::string output;
};

std::expected<Projection, Refusal> read(const OnnxView& g, const std::string& name) {
  const auto bad = [&](std::string_view what) {
    return refuse(RefusalCode::FieldMissing, "GMFlow : " + name + " : " + std::string(what));
  };
  const onnx::Node* mul = g.node(name + "/MatMul", "MatMul");
  const onnx::Node* add = g.node(name + "/Add", "Add");
  if (mul == nullptr || add == nullptr || mul->inputs.size() != 2 || add->inputs.size() != 2 ||
      mul->outputs.size() != 1 || add->outputs.size() != 1)
    return bad("MatMul suivi du biais attendu");
  const auto& product = mul->outputs[0];
  if (add->inputs[0] != product && add->inputs[1] != product) return bad("le biais ne lit pas le produit");
  const auto* w = g.constant(mul->inputs[1]);
  const auto* b = g.constant(add->inputs[add->inputs[0] == product ? 1U : 0U]);
  const auto fp16 = [](float x) { return std::isfinite(x) && std::abs(x) <= 65504.F; };
  if (w == nullptr || w->dims != std::vector<std::int64_t>{kD, kD} ||
      w->floats.size() != static_cast<std::size_t>(kD) * kD || !std::ranges::all_of(w->floats, fp16))
    return bad("matrice 128 × 128 dans le FP16 attendue");
  if (b == nullptr || b->dims != std::vector<std::int64_t>{kD} || b->floats.size() != kD ||
      !std::ranges::all_of(b->floats, fp16))
    return bad("biais à 128 valeurs finies attendu");

  Projection p{
      .data = {.weights = {}, .bias = b->floats}, .input = mul->inputs[0], .output = add->outputs[0]};
  p.data.weights.reserve(static_cast<std::size_t>(kD) * kD);
  for (std::size_t o = 0; o < kD; ++o)
    for (std::size_t i = 0; i < kD; ++i) p.data.weights.push_back(w->floats[i * kD + o]);
  return p;
}

}  // namespace

std::expected<GmflowPropagation, Refusal> gmflow_propagation(std::span<const std::byte> pair) {
  auto parsed = onnx::read_onnx(pair);
  if (!parsed) return refuse(RefusalCode::ModelUnreadable, parsed.error().detail);
  const OnnxView g(*parsed);
  auto q = read(g, std::string(kPrefix) + "q_proj");
  if (!q) return std::unexpected(std::move(q.error()));
  auto k = read(g, std::string(kPrefix) + "k_proj");
  if (!k) return std::unexpected(std::move(k.error()));
  auto q4 = read(g, std::string(kPrefix) + "q_proj_1");
  if (!q4) return std::unexpected(std::move(q4.error()));
  auto k4 = read(g, std::string(kPrefix) + "k_proj_1");
  if (!k4) return std::unexpected(std::move(k4.error()));
  if (k->input != q->output || k4->input != q4->input || q->data != q4->data || k->data != k4->data)
    return refuse(RefusalCode::FieldMissing,
                  "GMFlow : les projections de propagation diffèrent entre échelles");
  return GmflowPropagation{std::move(q->data), std::move(k->data)};
}

}  // namespace kirari::prepare
