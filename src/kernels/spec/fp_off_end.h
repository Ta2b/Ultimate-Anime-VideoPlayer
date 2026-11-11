// Referme la région ouverte par fp_off_begin.h.
// NOLINTBEGIN(llvm-header-guard)
#if defined(__CUDACC__)
// nvcc : les ports CUDA suivent leurs propres réglages.
#elif defined(__clang__)
#pragma float_control(pop)
#endif
// NOLINTEND(llvm-header-guard)
