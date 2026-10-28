// Plugin TensorRT 11 (IPluginV3) « AjpSpanTailFp8 » : remplace Add(saut) + Tanh + Mul + Q dans un
// bloc SPAN FP8. Entrées : 0 = c3 (FP8, sortie quantifiée de c3_r), 1 = att (FP16, sortie de
// att_conv avant le Tanh), 2 = saut (FP8) ; sortie : FP8 E4M3. Les échelles de déquantification
// viennent des attributs ONNX (s_c3, s_skip) et de l'échelle de la sortie (s_out), calibrées
// hors ligne. Prototype de banc : jamais chargé par le lecteur. Chargé par
// trtexec --staticPlugins=<libajp_span_tail_plugin.so>.
#include <NvInferRuntime.h>
#include <cstring>
#include <memory>

#include "../../../../src/gpu/cuda/backend/fp8_fused.cuh"

namespace {

using namespace nvinfer1;

constexpr char kName[] = "AjpSpanTailFp8";
constexpr char kVersion[] = "1";
constexpr char kNamespace[] = "";

class SpanTailPlugin final : public IPluginV3,
                             public IPluginV3OneCore,
                             public IPluginV3OneBuild,
                             public IPluginV3OneRuntime {
 public:
  SpanTailPlugin(float s_c3, float s_skip, float s_out) : s_c3_(s_c3), s_skip_(s_skip), s_out_(s_out) {
    fields_[0] = PluginField("s_c3", &s_c3_, PluginFieldType::kFLOAT32, 1);
    fields_[1] = PluginField("s_skip", &s_skip_, PluginFieldType::kFLOAT32, 1);
    fields_[2] = PluginField("s_out", &s_out_, PluginFieldType::kFLOAT32, 1);
    fc_.nbFields = 3;
    fc_.fields = fields_;
  }
  SpanTailPlugin(const SpanTailPlugin& o) : SpanTailPlugin(o.s_c3_, o.s_skip_, o.s_out_) {}

  // IPluginV3
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kCORE:
        return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD:
        return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME:
        return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  IPluginV3* clone() noexcept override { return new SpanTailPlugin(*this); }

  // IPluginV3OneCore
  const char* getPluginName() const noexcept override { return kName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const char* getPluginNamespace() const noexcept override { return kNamespace; }

  // IPluginV3OneBuild
  int32_t configurePlugin(const DynamicPluginTensorDesc*, int32_t, const DynamicPluginTensorDesc*,
                          int32_t) noexcept override {
    return 0;
  }
  int32_t getOutputDataTypes(DataType* out, int32_t, const DataType*, int32_t) const noexcept override {
    out[0] = DataType::kFP8;
    return 0;
  }
  int32_t getOutputShapes(const DimsExprs* in, int32_t, const DimsExprs*, int32_t, DimsExprs* out, int32_t,
                          IExprBuilder&) noexcept override {
    out[0] = in[0];
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, const DynamicPluginTensorDesc* io, int32_t,
                                 int32_t) noexcept override {
    const PluginTensorDesc& d = io[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    return pos == 1 ? d.type == DataType::kHALF : d.type == DataType::kFP8;
  }
  int32_t getNbOutputs() const noexcept override { return 1; }

  // IPluginV3OneRuntime
  int32_t onShapeChange(const PluginTensorDesc* in, int32_t, const PluginTensorDesc*,
                        int32_t) noexcept override {
    n_ = 1;
    for (int i = 0; i < in[0].dims.nbDims; ++i) n_ *= static_cast<size_t>(in[0].dims.d[i]);
    return (n_ % 16 == 0) ? 0 : 1;
  }
  int32_t enqueue(const PluginTensorDesc* in, const PluginTensorDesc*, const void* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    size_t n = 1;
    for (int i = 0; i < in[0].dims.nbDims; ++i) n *= static_cast<size_t>(in[0].dims.d[i]);
    ajp::gpu::cuda::Fp8SpanTailScales sc{s_c3_, s_skip_, 1.0f / s_out_};
    cudaError_t e = ajp::gpu::cuda::launch_fp8_span_tail(
        static_cast<const uint8_t*>(inputs[0]), static_cast<const __half*>(inputs[1]),
        static_cast<const uint8_t*>(inputs[2]), static_cast<uint8_t*>(outputs[0]), n, sc, true, stream);
    return e == cudaSuccess ? 0 : 1;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  const PluginFieldCollection* getFieldsToSerialize() noexcept override { return &fc_; }

 private:
  float s_c3_, s_skip_, s_out_;
  size_t n_ = 0;
  PluginField fields_[3];
  PluginFieldCollection fc_{};
};

class SpanTailCreator final : public IPluginCreatorV3One {
 public:
  SpanTailCreator() {
    fields_[0] = PluginField("s_c3", nullptr, PluginFieldType::kFLOAT32, 1);
    fields_[1] = PluginField("s_skip", nullptr, PluginFieldType::kFLOAT32, 1);
    fields_[2] = PluginField("s_out", nullptr, PluginFieldType::kFLOAT32, 1);
    fc_.nbFields = 3;
    fc_.fields = fields_;
  }
  IPluginV3* createPlugin(const char*, const PluginFieldCollection* fc, TensorRTPhase) noexcept override {
    float v[3] = {1.f, 1.f, 1.f};
    for (int i = 0; i < fc->nbFields; ++i) {
      const PluginField& f = fc->fields[i];
      if (f.data == nullptr || f.length < 1) continue;
      float x = f.type == PluginFieldType::kFLOAT64 ? static_cast<float>(*static_cast<const double*>(f.data))
                                                    : *static_cast<const float*>(f.data);
      if (!std::strcmp(f.name, "s_c3"))
        v[0] = x;
      else if (!std::strcmp(f.name, "s_skip"))
        v[1] = x;
      else if (!std::strcmp(f.name, "s_out"))
        v[2] = x;
    }
    return new SpanTailPlugin(v[0], v[1], v[2]);
  }
  const PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }
  const char* getPluginName() const noexcept override { return kName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const char* getPluginNamespace() const noexcept override { return kNamespace; }

 private:
  PluginField fields_[3];
  PluginFieldCollection fc_{};
};

REGISTER_TENSORRT_PLUGIN(SpanTailCreator);

}  // namespace
