# GENERATED FILE: DO NOT EDIT.
# Generator: loom.gen.target.capability_presets.
# Regenerate: python3 loom/py/loom/gen/run.py capability_presets --in-place
# Data only: loom_project.cmake owns preset resolution.

set(_LOOM_CAPABILITY_ARCHITECTURES
  "aie2p"
  "amdgpu"
  "spirv"
  "vm"
  "wasm"
  "x86"
)
set(_LOOM_CAPABILITY_ARTIFACT_FORMATS
  "elf"
  "spirv"
  "vm"
  "wasm"
)
set(_LOOM_CAPABILITY_EXECUTION_SUBSTRATES
  "iree_hal"
  "iree_vm"
  "wasm_host"
)
set(_LOOM_TARGET_PRESET_KEYS
  "VM"
  "CPU"
  "AMD_GPU"
  "AMD_NPU"
  "AMD"
  "SPIRV"
  "WASM"
  "DEFAULT"
  "ALL"
)
set(_LOOM_TARGET_FAMILY_PRESET_KEYS
  "VM"
  "CPU"
  "AMD_GPU"
  "AMD_NPU"
  "AMD"
  "SPIRV"
  "WASM"
  "ALL"
)

set(_LOOM_TARGET_PRESET_VM_OPTION "LOOM_TARGET_ALL_VM")
set(_LOOM_TARGET_PRESET_VM_DESCRIPTION "Build Loom VM compilation and execution support.")
set(_LOOM_TARGET_PRESET_VM_ARCHITECTURES
  "vm"
)
set(_LOOM_TARGET_PRESET_VM_ARTIFACTS
  "vm"
)
set(_LOOM_TARGET_PRESET_VM_EXECUTION
  "iree_vm"
)

set(_LOOM_TARGET_PRESET_CPU_OPTION "LOOM_TARGET_ALL_CPU")
set(_LOOM_TARGET_PRESET_CPU_DESCRIPTION "Build Loom CPU and supporting VM capabilities.")
set(_LOOM_TARGET_PRESET_CPU_ARCHITECTURES
  "vm"
  "x86"
)
set(_LOOM_TARGET_PRESET_CPU_ARTIFACTS
  "elf"
  "vm"
)
set(_LOOM_TARGET_PRESET_CPU_EXECUTION
  "iree_hal"
  "iree_vm"
)

set(_LOOM_TARGET_PRESET_AMD_GPU_OPTION "LOOM_TARGET_ALL_AMD_GPU")
set(_LOOM_TARGET_PRESET_AMD_GPU_DESCRIPTION "Build Loom AMD GPU and supporting VM capabilities.")
set(_LOOM_TARGET_PRESET_AMD_GPU_ARCHITECTURES
  "amdgpu"
  "vm"
)
set(_LOOM_TARGET_PRESET_AMD_GPU_ARTIFACTS
  "elf"
  "vm"
)
set(_LOOM_TARGET_PRESET_AMD_GPU_EXECUTION
  "iree_hal"
  "iree_vm"
)

set(_LOOM_TARGET_PRESET_AMD_NPU_OPTION "LOOM_TARGET_ALL_AMD_NPU")
set(_LOOM_TARGET_PRESET_AMD_NPU_DESCRIPTION "Build Loom AMD NPU and supporting VM capabilities.")
set(_LOOM_TARGET_PRESET_AMD_NPU_ARCHITECTURES
  "aie2p"
  "vm"
)
set(_LOOM_TARGET_PRESET_AMD_NPU_ARTIFACTS
  "elf"
  "vm"
)
set(_LOOM_TARGET_PRESET_AMD_NPU_EXECUTION
  "iree_hal"
  "iree_vm"
)

set(_LOOM_TARGET_PRESET_AMD_OPTION "LOOM_TARGET_ALL_AMD")
set(_LOOM_TARGET_PRESET_AMD_DESCRIPTION "Build Loom AMD GPU, NPU, and supporting VM capabilities.")
set(_LOOM_TARGET_PRESET_AMD_ARCHITECTURES
  "aie2p"
  "amdgpu"
  "vm"
)
set(_LOOM_TARGET_PRESET_AMD_ARTIFACTS
  "elf"
  "vm"
)
set(_LOOM_TARGET_PRESET_AMD_EXECUTION
  "iree_hal"
  "iree_vm"
)

set(_LOOM_TARGET_PRESET_SPIRV_OPTION "LOOM_TARGET_ALL_SPIRV")
set(_LOOM_TARGET_PRESET_SPIRV_DESCRIPTION "Build Loom SPIR-V and supporting VM capabilities.")
set(_LOOM_TARGET_PRESET_SPIRV_ARCHITECTURES
  "spirv"
  "vm"
)
set(_LOOM_TARGET_PRESET_SPIRV_ARTIFACTS
  "spirv"
  "vm"
)
set(_LOOM_TARGET_PRESET_SPIRV_EXECUTION
  "iree_hal"
  "iree_vm"
)

set(_LOOM_TARGET_PRESET_WASM_OPTION "LOOM_TARGET_ALL_WASM")
set(_LOOM_TARGET_PRESET_WASM_DESCRIPTION "Build Loom WebAssembly and supporting VM capabilities.")
set(_LOOM_TARGET_PRESET_WASM_ARCHITECTURES
  "vm"
  "wasm"
)
set(_LOOM_TARGET_PRESET_WASM_ARTIFACTS
  "vm"
  "wasm"
)
set(_LOOM_TARGET_PRESET_WASM_EXECUTION
  "iree_vm"
  "wasm_host"
)

set(_LOOM_TARGET_PRESET_DEFAULT_OPTION "LOOM_TARGET_DEFAULT")
set(_LOOM_TARGET_PRESET_DEFAULT_DESCRIPTION "Build the ordinary non-WebAssembly Loom target set.")
set(_LOOM_TARGET_PRESET_DEFAULT_ARCHITECTURES
  "aie2p"
  "amdgpu"
  "spirv"
  "vm"
  "x86"
)
set(_LOOM_TARGET_PRESET_DEFAULT_ARTIFACTS
  "elf"
  "spirv"
  "vm"
)
set(_LOOM_TARGET_PRESET_DEFAULT_EXECUTION
  "iree_hal"
  "iree_vm"
)

set(_LOOM_TARGET_PRESET_ALL_OPTION "LOOM_TARGET_ALL")
set(_LOOM_TARGET_PRESET_ALL_DESCRIPTION "Build every Loom target capability.")
set(_LOOM_TARGET_PRESET_ALL_ARCHITECTURES
  "aie2p"
  "amdgpu"
  "spirv"
  "vm"
  "wasm"
  "x86"
)
set(_LOOM_TARGET_PRESET_ALL_ARTIFACTS
  "elf"
  "spirv"
  "vm"
  "wasm"
)
set(_LOOM_TARGET_PRESET_ALL_EXECUTION
  "iree_hal"
  "iree_vm"
  "wasm_host"
)
