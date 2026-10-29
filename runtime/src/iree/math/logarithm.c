// Copyright (c) 2017-2025, Arm Limited.
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/logarithm.h"

#include <stdbool.h>

#include "iree/math/float_bits.h"
#include "iree/math/logarithm_reduction.h"

// The table-and-polynomial kernels derive from Arm Optimized Routines
// 2ee4b0d45680a9880412c1d186224b08652c35f6 with fixed unfused evaluation.
// The frozen log2 approximation retains its finite mapping and flush shell.

//===----------------------------------------------------------------------===//
// Base-two logarithm
//===----------------------------------------------------------------------===//

const iree_math_log2_f32_table_entry_t iree_math_log2_f32_table[16] = {
    {0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2},
    {0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2},
    {0x1.49539f0f010bp+0, -0x1.7418b0a1fb77bp-2},
    {0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2},
    {0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2},
    {0x1.25e227b0b8eap+0, -0x1.97c1d1b3b7afp-3},
    {0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3},
    {0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4},
    {0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5},
    {0x1p+0, 0x0p+0},
    {0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4},
    {0x1.ca4b31f026aap-1, 0x1.476a9543891bap-3},
    {0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3},
    {0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2},
    {0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2},
    {0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2},
};

static const double iree_math_log2_f32_polynomial[4] = {
    -0x1.712b6f70a7e4dp-2,
    0x1.ecabf496832ep-2,
    -0x1.715479ffae3dep-1,
    0x1.715475f35c8b8p0,
};

static bool iree_math_log2_f32_handle_special(float value, float* out_result) {
  const uint32_t value_bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude_bits = value_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude_bits < IREE_MATH_F32_MIN_NORMAL)) {
    *out_result = iree_math_f32_from_bits(UINT32_C(0xFF800000));
    return true;
  }
  if (IREE_UNLIKELY((value_bits & IREE_MATH_F32_SIGN_BIT) != 0)) {
    *out_result = iree_math_f32_canonical_nan();
    return true;
  }
  if (IREE_UNLIKELY(magnitude_bits >= IREE_MATH_F32_INFINITY)) {
    *out_result = magnitude_bits == IREE_MATH_F32_INFINITY
                      ? iree_math_f32_from_bits(IREE_MATH_F32_INFINITY)
                      : iree_math_f32_canonical_nan();
    return true;
  }
  return false;
}

static float iree_math_log2_f32_finite(uint32_t value_bits) {
  // Normalize the input and select one of sixteen centers without a division.
  const uint32_t adjusted_bits = value_bits - UINT32_C(0x3F330000);
  const uint32_t table_index = (adjusted_bits >> 19) & UINT32_C(15);
  const uint32_t exponent_code = adjusted_bits >> 23;
  const int32_t exponent = exponent_code >= UINT32_C(256)
                               ? (int32_t)exponent_code - 512
                               : (int32_t)exponent_code;
  const uint32_t normalized_bits =
      value_bits - (adjusted_bits & UINT32_C(0xFF800000));
  const double normalized = (double)iree_math_f32_from_bits(normalized_bits);
  const double residual =
      normalized * iree_math_log2_f32_table[table_index].inverse_center - 1.0;
  const double base =
      iree_math_log2_f32_table[table_index].log2_center + (double)exponent;

  const double residual_squared = residual * residual;
  double upper = iree_math_log2_f32_polynomial[1] * residual +
                 iree_math_log2_f32_polynomial[2];
  upper = iree_math_log2_f32_polynomial[0] * residual_squared + upper;
  const double lower = iree_math_log2_f32_polynomial[3] * residual + base;
  return (float)(upper * residual_squared + lower);
}

IREE_API_EXPORT float iree_math_log2_f32_approx(float value) {
  float result = 0.0f;
  if (iree_math_log2_f32_handle_special(value, &result)) {
    return result;
  }
  return iree_math_log2_f32_finite(iree_math_f32_to_bits(value));
}

// The split center retains a single-reduction-error residual without FMA.
// Natural and base-two logarithms use different qualified reduction grids.
typedef struct iree_math_log_f64_table_entry_t {
  // Reciprocal of the subinterval center.
  double inverse_center;
  // Logarithm of the center in the table's base.
  double log_center;
  // Leading center value for unfused range reduction.
  double center_high;
  // Trailing center value for unfused range reduction.
  double center_low;
} iree_math_log_f64_table_entry_t;

static const iree_math_log_f64_table_entry_t iree_math_log_f64_table[128] = {
    {0x1.734f0c3e0de9fp+0, -0x1.7cc7f79e69000p-2, 0x1.61000014fb66bp-1,
     0x1.e026c91425b3cp-56},
    {0x1.713786a2ce91fp+0, -0x1.76feec20d0000p-2, 0x1.63000034db495p-1,
     0x1.dbfea48005d41p-55},
    {0x1.6f26008fab5a0p+0, -0x1.713e31351e000p-2, 0x1.650000d94d478p-1,
     0x1.e7fa786d6a5b7p-55},
    {0x1.6d1a61f138c7dp+0, -0x1.6b85b38287800p-2, 0x1.67000074e6fadp-1,
     0x1.1fcea6b54254cp-57},
    {0x1.6b1490bc5b4d1p+0, -0x1.65d5590807800p-2, 0x1.68ffffedf0faep-1,
     -0x1.c7e274c590efdp-56},
    {0x1.69147332f0cbap+0, -0x1.602d076180000p-2, 0x1.6b0000763c5bcp-1,
     -0x1.ac16848dcda01p-55},
    {0x1.6719f18224223p+0, -0x1.5a8ca86909000p-2, 0x1.6d0001e5cc1f6p-1,
     0x1.33f1c9d499311p-55},
    {0x1.6524f99a51ed9p+0, -0x1.54f4356035000p-2, 0x1.6efffeb05f63ep-1,
     -0x1.e80041ae22d53p-56},
    {0x1.63356aa8f24c4p+0, -0x1.4f637c36b4000p-2, 0x1.710000e86978p-1,
     0x1.bff6671097952p-56},
    {0x1.614b36b9ddc14p+0, -0x1.49da7fda85000p-2, 0x1.72ffffc67e912p-1,
     0x1.c00e226bd8724p-55},
    {0x1.5f66452c65c4cp+0, -0x1.445923989a800p-2, 0x1.74fffdf81116ap-1,
     -0x1.e02916ef101d2p-57},
    {0x1.5d867b5912c4fp+0, -0x1.3edf439b0b800p-2, 0x1.770000f679c9p-1,
     -0x1.7fc71cd549c74p-57},
    {0x1.5babccb5b90dep+0, -0x1.396ce448f7000p-2, 0x1.78ffffa7ec835p-1,
     0x1.1bec19ef50483p-55},
    {0x1.59d61f2d91a78p+0, -0x1.3401e17bda000p-2, 0x1.7affffe20c2e6p-1,
     -0x1.07e1729cc6465p-56},
    {0x1.5805612465687p+0, -0x1.2e9e2ef468000p-2, 0x1.7cfffed3fc9p-1,
     -0x1.08072087b8b1cp-55},
    {0x1.56397cee76bd3p+0, -0x1.2941b3830e000p-2, 0x1.7efffe9261a76p-1,
     0x1.dc0286d9df9aep-55},
    {0x1.54725e2a77f93p+0, -0x1.23ec58cda8800p-2, 0x1.81000049ca3e8p-1,
     0x1.97fd251e54c33p-55},
    {0x1.52aff42064583p+0, -0x1.1e9e129279000p-2, 0x1.8300017932c8fp-1,
     -0x1.afee9b630f381p-55},
    {0x1.50f22dbb2bddfp+0, -0x1.1956d2b48f800p-2, 0x1.850000633739cp-1,
     0x1.9bfbf6b6535bcp-55},
    {0x1.4f38f4734ded7p+0, -0x1.141679ab9f800p-2, 0x1.87000204289c6p-1,
     -0x1.bbf65f3117b75p-55},
    {0x1.4d843cfde2840p+0, -0x1.0edd094ef9800p-2, 0x1.88fffebf57904p-1,
     -0x1.9006ea23dcb57p-55},
    {0x1.4bd3ec078a3c8p+0, -0x1.09aa518db1000p-2, 0x1.8b00022bc04dfp-1,
     -0x1.d00df38e04b0ap-56},
    {0x1.4a27fc3e0258ap+0, -0x1.047e65263b800p-2, 0x1.8cfffe50c1b8ap-1,
     -0x1.8007146ff9f05p-55},
    {0x1.4880524d48434p+0, -0x1.feb224586f000p-3, 0x1.8effffc918e43p-1,
     0x1.3817bd07a7038p-55},
    {0x1.46dce1b192d0bp+0, -0x1.f474a7517b000p-3, 0x1.910001efa5fc7p-1,
     0x1.93e9176dfb403p-55},
    {0x1.453d9d3391854p+0, -0x1.ea4443d103000p-3, 0x1.9300013467bb9p-1,
     0x1.f804e4b980276p-56},
    {0x1.43a2744b4845ap+0, -0x1.e020d44e9b000p-3, 0x1.94fffe6ee076fp-1,
     -0x1.f7ef0d9ff622ep-55},
    {0x1.420b54115f8fbp+0, -0x1.d60a22977f000p-3, 0x1.96fffde3c12d1p-1,
     -0x1.082aa962638bap-56},
    {0x1.40782da3ef4b1p+0, -0x1.cc00104959000p-3, 0x1.98ffff4458a0dp-1,
     -0x1.7801b9164a8efp-55},
    {0x1.3ee8f5d57fe8fp+0, -0x1.c202956891000p-3, 0x1.9afffdd982e3ep-1,
     -0x1.740e08a5a9337p-55},
    {0x1.3d5d9a00b4ce9p+0, -0x1.b81178d811000p-3, 0x1.9cfffed49fb66p-1,
     0x1.fce08c19bep-60},
    {0x1.3bd60c010c12bp+0, -0x1.ae2c9ccd3d000p-3, 0x1.9f00020f19c51p-1,
     -0x1.a3faa27885b0ap-55},
    {0x1.3a5242b75dab8p+0, -0x1.a45402e129000p-3, 0x1.a10001145b006p-1,
     0x1.4ff489958da56p-56},
    {0x1.38d22cd9fd002p+0, -0x1.9a877681df000p-3, 0x1.a300007bbf6fap-1,
     0x1.cbeab8a2b6d18p-55},
    {0x1.3755bc5847a1cp+0, -0x1.90c6d69483000p-3, 0x1.a500010971d79p-1,
     0x1.8fecadd78793p-55},
    {0x1.35dce49ad36e2p+0, -0x1.87120a645c000p-3, 0x1.a70001df52e48p-1,
     -0x1.f41763dd8abdbp-55},
    {0x1.34679984dd440p+0, -0x1.7d68fb4143000p-3, 0x1.a90001c593352p-1,
     -0x1.ebf0284c27612p-55},
    {0x1.32f5cceffcb24p+0, -0x1.73cb83c627000p-3, 0x1.ab0002a4f3e4bp-1,
     -0x1.9fd043cff3f5fp-57},
    {0x1.3187775a10d49p+0, -0x1.6a39a9b376000p-3, 0x1.acfffd7ae1ed1p-1,
     -0x1.23ee7129070b4p-55},
    {0x1.301c8373e3990p+0, -0x1.60b3154b7a000p-3, 0x1.aefffee510478p-1,
     0x1.a063ee00edea3p-57},
    {0x1.2eb4ebb95f841p+0, -0x1.5737d76243000p-3, 0x1.b0fffdb650d5bp-1,
     0x1.a06c8381f0ab9p-58},
    {0x1.2d50a0219a9d1p+0, -0x1.4dc7b8fc23000p-3, 0x1.b2ffffeaaca57p-1,
     -0x1.9011e74233c1dp-56},
    {0x1.2bef9a8b7fd2ap+0, -0x1.4462c51d20000p-3, 0x1.b4fffd995badcp-1,
     -0x1.9ff1068862a9fp-56},
    {0x1.2a91c7a0c1babp+0, -0x1.3b08abc830000p-3, 0x1.b7000249e659cp-1,
     0x1.aff45d0864f3ep-55},
    {0x1.293726014b530p+0, -0x1.31b996b490000p-3, 0x1.b8ffff987164p-1,
     0x1.cfe7796c2c3f9p-56},
    {0x1.27dfa5757a1f5p+0, -0x1.2875490a44000p-3, 0x1.bafffd204cb4fp-1,
     -0x1.3ff27eef22bc4p-57},
    {0x1.268b39b1d3bbfp+0, -0x1.1f3b9f879a000p-3, 0x1.bcfffd2415c45p-1,
     -0x1.cffb7ee3bea21p-57},
    {0x1.2539d838ff5bdp+0, -0x1.160c8252ca000p-3, 0x1.beffff86309dfp-1,
     -0x1.14103972e0b5cp-55},
    {0x1.23eb7aac9083bp+0, -0x1.0ce7f57f72000p-3, 0x1.c0fffe1b57653p-1,
     0x1.bc16494b76a19p-55},
    {0x1.22a012ba940b6p+0, -0x1.03cdc49fea000p-3, 0x1.c2ffff1fa57e3p-1,
     -0x1.4feef8d30c6edp-57},
    {0x1.2157996cc4132p+0, -0x1.f57bdbc4b8000p-4, 0x1.c4fffdcbfe424p-1,
     -0x1.43f68bcec4775p-55},
    {0x1.201201dd2fc9bp+0, -0x1.e370896404000p-4, 0x1.c6fffed54b9f7p-1,
     0x1.47ea3f053e0ecp-55},
    {0x1.1ecf4494d480bp+0, -0x1.d17983ef94000p-4, 0x1.c8fffeb998fd5p-1,
     0x1.383068df992f1p-56},
    {0x1.1d8f5528f6569p+0, -0x1.bf9674ed8a000p-4, 0x1.cb0002125219ap-1,
     -0x1.8fd8e64180e04p-57},
    {0x1.1c52311577e7cp+0, -0x1.adc79202f6000p-4, 0x1.ccfffdd94469cp-1,
     0x1.e7ebe1cc7ea72p-55},
    {0x1.1b17c74cb26e9p+0, -0x1.9c0c3e7288000p-4, 0x1.cefffeafdc476p-1,
     0x1.ebe39ad9f88fep-55},
    {0x1.19e010c2c1ab6p+0, -0x1.8a646b372c000p-4, 0x1.d1000169af82bp-1,
     0x1.57d91a8b95a71p-56},
    {0x1.18ab07bb670bdp+0, -0x1.78d01b3ac0000p-4, 0x1.d30000d0ff71dp-1,
     0x1.9c1906970c7dap-55},
    {0x1.1778a25efbcb6p+0, -0x1.674f145380000p-4, 0x1.d4fffea790fc4p-1,
     -0x1.80e37c558fe0cp-58},
    {0x1.1648d354c31dap+0, -0x1.55e0e6d878000p-4, 0x1.d70002edc87e5p-1,
     -0x1.f80d64dc10f44p-56},
    {0x1.151b990275fddp+0, -0x1.4485cdea1e000p-4, 0x1.d900021dc82aap-1,
     -0x1.47c8f94fd5c5cp-56},
    {0x1.13f0ea432d24cp+0, -0x1.333d94d6aa000p-4, 0x1.dafffd86b0283p-1,
     0x1.c7f1dc521617ep-55},
    {0x1.12c8b7210f9dap+0, -0x1.22079f8c56000p-4, 0x1.dd000296c4739p-1,
     0x1.8019eb2ffb153p-55},
    {0x1.11a3028ecb531p+0, -0x1.10e4698622000p-4, 0x1.defffe54490f5p-1,
     0x1.e00d2c652cc89p-57},
    {0x1.107fbda8434afp+0, -0x1.ffa6c6ad20000p-5, 0x1.e0fffcdabf694p-1,
     -0x1.f8340202d69d2p-56},
    {0x1.0f5ee0f4e6bb3p+0, -0x1.dda8d4a774000p-5, 0x1.e2fffdb52c8ddp-1,
     0x1.b00c1ca1b0864p-56},
    {0x1.0e4065d2a9fcep+0, -0x1.bbcece4850000p-5, 0x1.e4ffff24216efp-1,
     0x1.2ffa8b094ab51p-56},
    {0x1.0d244632ca521p+0, -0x1.9a1894012c000p-5, 0x1.e6fffe88a5e11p-1,
     -0x1.7f673b1efbe59p-58},
    {0x1.0c0a77ce2981ap+0, -0x1.788583302c000p-5, 0x1.e9000119eff0dp-1,
     -0x1.4808d5e0bc801p-55},
    {0x1.0af2f83c636d1p+0, -0x1.5715e67d68000p-5, 0x1.eafffdfa51744p-1,
     0x1.80006d54320b5p-56},
    {0x1.09ddb98a01339p+0, -0x1.35c8a49658000p-5, 0x1.ed0001a127fa1p-1,
     -0x1.002f860565c92p-58},
    {0x1.08cabaf52e7dfp+0, -0x1.149e364154000p-5, 0x1.ef00007babcc4p-1,
     -0x1.540445d35e611p-55},
    {0x1.07b9f2f4e28fbp+0, -0x1.e72c082eb8000p-6, 0x1.f0ffff57a8d02p-1,
     -0x1.ffb3139ef9105p-59},
    {0x1.06ab58c358f19p+0, -0x1.a55f152528000p-6, 0x1.f30001ee58ac7p-1,
     0x1.a81acf2731155p-55},
    {0x1.059eea5ecf92cp+0, -0x1.63d62cf818000p-6, 0x1.f4ffff5823494p-1,
     0x1.a3f41d4d7c743p-55},
    {0x1.04949cdd12c90p+0, -0x1.228fb8caa0000p-6, 0x1.f6ffffca94c6bp-1,
     -0x1.202f41c987875p-57},
    {0x1.038c6c6f0ada9p+0, -0x1.c317b20f90000p-7, 0x1.f8fffe1f9c441p-1,
     0x1.77dd1f477e74bp-56},
    {0x1.02865137932a9p+0, -0x1.419355daa0000p-7, 0x1.fafffd2e0e37ep-1,
     -0x1.f01199a7ca331p-57},
    {0x1.0182427ea7348p+0, -0x1.81203c2ec0000p-8, 0x1.fd0001c77e49ep-1,
     0x1.181ee4bceacb1p-56},
    {0x1.008040614b195p+0, -0x1.0040979240000p-9, 0x1.feffff7e0c331p-1,
     -0x1.e05370170875ap-57},
    {0x1.fe01ff726fa1ap-1, 0x1.feff384900000p-9, 0x1.00ffff465606ep+0,
     -0x1.a7ead491c0adap-55},
    {0x1.fa11cc261ea74p-1, 0x1.7dc41353d0000p-7, 0x1.02ffff3867a58p+0,
     -0x1.77f69c3fcb2ep-54},
    {0x1.f6310b081992ep-1, 0x1.3cea3c4c28000p-6, 0x1.04ffffdfc0d17p+0,
     0x1.7bffe34cb945bp-54},
    {0x1.f25f63ceeadcdp-1, 0x1.b9fc114890000p-6, 0x1.0700003cd4d82p+0,
     0x1.20083c0e456cbp-55},
    {0x1.ee9c8039113e7p-1, 0x1.1b0d8ce110000p-5, 0x1.08ffff9f2cbe8p+0,
     -0x1.dffdfbe37751ap-57},
    {0x1.eae8078cbb1abp-1, 0x1.58a5bd001c000p-5, 0x1.0b000010cda65p+0,
     -0x1.13f7faee626ebp-54},
    {0x1.e741aa29d0c9bp-1, 0x1.95c8340d88000p-5, 0x1.0d00001a4d338p+0,
     0x1.07dfa79489ff7p-55},
    {0x1.e3a91830a99b5p-1, 0x1.d276aef578000p-5, 0x1.0effffadafdfdp+0,
     -0x1.7040570d66bcp-56},
    {0x1.e01e009609a56p-1, 0x1.07598e598c000p-4, 0x1.110000bbafd96p+0,
     0x1.e80d4846d0b62p-55},
    {0x1.dca01e577bb98p-1, 0x1.253f5e30d2000p-4, 0x1.12ffffae5f45dp+0,
     0x1.dbffa64fd36efp-54},
    {0x1.d92f20b7c9103p-1, 0x1.42edd8b380000p-4, 0x1.150000dd59ad9p+0,
     0x1.a0077701250aep-54},
    {0x1.d5cac66fb5ccep-1, 0x1.606598757c000p-4, 0x1.170000f21559ap+0,
     0x1.dfdf9e2e3deeep-55},
    {0x1.d272caa5ede9dp-1, 0x1.7da76356a0000p-4, 0x1.18ffffc275426p+0,
     0x1.10030dc3b7273p-54},
    {0x1.cf26e3e6b2ccdp-1, 0x1.9ab434e1c6000p-4, 0x1.1b000123d3c59p+0,
     0x1.97f7980030188p-54},
    {0x1.cbe6da2a77902p-1, 0x1.b78c7bb0d6000p-4, 0x1.1cffff8299eb7p+0,
     -0x1.5f932ab9f8c67p-57},
    {0x1.c8b266d37086dp-1, 0x1.d431332e72000p-4, 0x1.1effff48ad4p+0,
     0x1.37fbf9da75bebp-54},
    {0x1.c5894bd5d5804p-1, 0x1.f0a3171de6000p-4, 0x1.210000c8b86a4p+0,
     0x1.f806b91fd5b22p-54},
    {0x1.c26b533bb9f8cp-1, 0x1.067152b914000p-3, 0x1.2300003854303p+0,
     0x1.3ffc2eb9fbf33p-54},
    {0x1.bf583eeece73fp-1, 0x1.147858292b000p-3, 0x1.24fffffbcf684p+0,
     0x1.601e77e2e2e72p-56},
    {0x1.bc4fd75db96c1p-1, 0x1.2266ecdca3000p-3, 0x1.26ffff52921d9p+0,
     0x1.ffcbb767f0c61p-56},
    {0x1.b951e0c864a28p-1, 0x1.303d7a6c55000p-3, 0x1.2900014933a3cp+0,
     -0x1.202ca3c02412bp-56},
    {0x1.b65e2c5ef3e2cp-1, 0x1.3dfc33c331000p-3, 0x1.2b00014556313p+0,
     -0x1.2808233f21f02p-54},
    {0x1.b374867c9888bp-1, 0x1.4ba366b7a8000p-3, 0x1.2cfffebfe523bp+0,
     -0x1.8ff7e384fdcf2p-55},
    {0x1.b094b211d304ap-1, 0x1.5933928d1f000p-3, 0x1.2f0000bb8ad96p+0,
     -0x1.5ff51503041c5p-55},
    {0x1.adbe885f2ef7ep-1, 0x1.66acd2418f000p-3, 0x1.30ffffb7ae2afp+0,
     -0x1.10071885e289dp-55},
    {0x1.aaf1d31603da2p-1, 0x1.740f8ec669000p-3, 0x1.32ffffeac5f7fp+0,
     -0x1.1ff5d3fb7b715p-54},
    {0x1.a82e63fd358a7p-1, 0x1.815c0f51af000p-3, 0x1.350000ca66756p+0,
     0x1.57f82228b82bdp-54},
    {0x1.a5740ef09738bp-1, 0x1.8e92954f68000p-3, 0x1.3700011fbf721p+0,
     0x1.000bac40dd5ccp-55},
    {0x1.a2c2a90ab4b27p-1, 0x1.9bb3602f84000p-3, 0x1.38ffff9592fb9p+0,
     -0x1.43f9d2db2a751p-54},
    {0x1.a01a01393f2d1p-1, 0x1.a8bed1c2c0000p-3, 0x1.3b00004ddd242p+0,
     0x1.57f6b707638e1p-55},
    {0x1.9d79f24db3c1bp-1, 0x1.b5b515c01d000p-3, 0x1.3cffff5b2c957p+0,
     0x1.a023a10bf1231p-56},
    {0x1.9ae2505c7b190p-1, 0x1.c2967ccbcc000p-3, 0x1.3efffeab0b418p+0,
     0x1.87f6d66b152bp-54},
    {0x1.9852ef297ce2fp-1, 0x1.cf635d5486000p-3, 0x1.410001532aff4p+0,
     0x1.7f8375f198524p-57},
    {0x1.95cbaeea44b75p-1, 0x1.dc1bd3446c000p-3, 0x1.4300017478b29p+0,
     0x1.301e672dc5143p-55},
    {0x1.934c69de74838p-1, 0x1.e8c01b8cfe000p-3, 0x1.44fffe795b463p+0,
     0x1.9ff69b8b2895ap-55},
    {0x1.90d4f2f6752e6p-1, 0x1.f5509c0179000p-3, 0x1.46fffe80475ep+0,
     -0x1.5c0b19bc2f254p-54},
    {0x1.8e6528effd79dp-1, 0x1.00e6c121fb800p-2, 0x1.48fffef6fc1e7p+0,
     0x1.b4009f23a2a72p-54},
    {0x1.8bfce9fcc007cp-1, 0x1.071b80e93d000p-2, 0x1.4afffe5bea704p+0,
     -0x1.4ffb7bf0d7d45p-54},
    {0x1.899c0dabec30ep-1, 0x1.0d46b9e867000p-2, 0x1.4d000171027dep+0,
     -0x1.9c06471dc6a3dp-54},
    {0x1.87427aa2317fbp-1, 0x1.13687334bd000p-2, 0x1.4f0000ff03ee2p+0,
     0x1.77f890b85531cp-54},
    {0x1.84f00acb39a08p-1, 0x1.1980d67234800p-2, 0x1.5100012dc4bd1p+0,
     0x1.004657166a436p-57},
    {0x1.82a49e8653e55p-1, 0x1.1f8ffe0cc8000p-2, 0x1.530001605277ap+0,
     -0x1.6bfcece233209p-54},
    {0x1.8060195f40260p-1, 0x1.2595fd7636800p-2, 0x1.54fffecdb704cp+0,
     -0x1.902720505a1d7p-55},
    {0x1.7e22563e0a329p-1, 0x1.2b9300914a800p-2, 0x1.56fffef5f54a9p+0,
     0x1.bbfe60ec96412p-54},
    {0x1.7beb377dcb5adp-1, 0x1.3187210436000p-2, 0x1.5900017e61012p+0,
     0x1.87ec581afef9p-55},
    {0x1.79baa679725c2p-1, 0x1.377266dec1800p-2, 0x1.5b00003c93e92p+0,
     -0x1.f41080abf0ccp-54},
    {0x1.77907f2170657p-1, 0x1.3d54ffbaf3000p-2, 0x1.5d0001d4919bcp+0,
     -0x1.8812afb254729p-54},
    {0x1.756cadbd6130cp-1, 0x1.432eee32fe000p-2, 0x1.5efffe7b87a89p+0,
     -0x1.47eb780ed6904p-54},
};

static const double iree_math_log_f64_polynomial[] = {
    -0x1.0000000000001p-1, 0x1.555555551305bp-2,  -0x1.fffffffeb459p-3,
    0x1.999b324f10111p-3,  -0x1.55575e506c89fp-3,
};

static const double iree_math_log_f64_near_one[] = {
    -0x1p-1,
    0x1.5555555555577p-2,
    -0x1.ffffffffffdcbp-3,
    0x1.999999995dd0cp-3,
    -0x1.55555556745a7p-3,
    0x1.24924a344de3p-3,
    -0x1.fffffa4423d65p-4,
    0x1.c7184282ad6cap-4,
    -0x1.999eb43b068ffp-4,
    0x1.78182f7afd085p-4,
    -0x1.5521375d145cdp-4,
};

static const iree_math_log_f64_table_entry_t iree_math_log2_f64_table[64] = {
    {0x1.724286bb1acf8p+0, -0x1.1095feecdb000p-1, 0x1.6200012b90a8ep-1,
     0x1.904ab0644b605p-55},
    {0x1.6e1f766d2cca1p+0, -0x1.08494bd76d000p-1, 0x1.66000045734a6p-1,
     0x1.1ff9bea62f7a9p-57},
    {0x1.6a13d0e30d48ap+0, -0x1.00143aee8f800p-1, 0x1.69fffc325f2c5p-1,
     0x1.27ecfcb3c90bap-55},
    {0x1.661ec32d06c85p+0, -0x1.efec5360b4000p-2, 0x1.6e00038b95a04p-1,
     0x1.8ff8856739326p-55},
    {0x1.623fa951198f8p+0, -0x1.dfdd91ab7e000p-2, 0x1.71fffe09994e3p-1,
     0x1.afd40275f82b1p-55},
    {0x1.5e75ba4cf026cp+0, -0x1.cffae0cc79000p-2, 0x1.7600015590e1p-1,
     -0x1.2fd75b4238341p-56},
    {0x1.5ac055a214fb8p+0, -0x1.c043811fda000p-2, 0x1.7a00012655bd5p-1,
     0x1.808e67c242b76p-56},
    {0x1.571ed0f166e1ep+0, -0x1.b0b67323ae000p-2, 0x1.7e0003259e9a6p-1,
     -0x1.208e426f622b7p-57},
    {0x1.53909590bf835p+0, -0x1.a152f5a2db000p-2, 0x1.81fffedb4b2d2p-1,
     -0x1.402461ea5c92fp-55},
    {0x1.5014fed61adddp+0, -0x1.9217f5af86000p-2, 0x1.860002dfafcc3p-1,
     0x1.df7f4a2f29a1fp-57},
    {0x1.4cab88e487bd0p+0, -0x1.8304db0719000p-2, 0x1.89ffff78c6b5p-1,
     -0x1.e0453094995fdp-55},
    {0x1.49539b4334feep+0, -0x1.74189f9a9e000p-2, 0x1.8e00039671566p-1,
     -0x1.a04f3bec77b45p-55},
    {0x1.460cbdfafd569p+0, -0x1.6552bb5199000p-2, 0x1.91fffe2bf1745p-1,
     -0x1.7fa34400e203cp-56},
    {0x1.42d664ee4b953p+0, -0x1.56b23a29b1000p-2, 0x1.95fffcc5c9fd1p-1,
     -0x1.6ff8005a0695dp-56},
    {0x1.3fb01111dd8a6p+0, -0x1.483650f5fa000p-2, 0x1.9a0003bba4767p-1,
     0x1.0f8c4c4ec7e03p-56},
    {0x1.3c995b70c5836p+0, -0x1.39de937f6a000p-2, 0x1.9dfffe7b92da5p-1,
     0x1.e7fd9478c4602p-55},
    {0x1.3991c4ab6fd4ap+0, -0x1.2baa1538d6000p-2, 0x1.a1fffd72efdafp-1,
     -0x1.a0c554dcdae7ep-57},
    {0x1.3698e0ce099b5p+0, -0x1.1d98340ca4000p-2, 0x1.a5fffde04ff95p-1,
     0x1.67da98ce9b26bp-55},
    {0x1.33ae48213e7b2p+0, -0x1.0fa853a40e000p-2, 0x1.a9fffca5e8d2bp-1,
     -0x1.284c9b54c13dep-55},
    {0x1.30d191985bdb1p+0, -0x1.01d9c32e73000p-2, 0x1.adfffddad03eap-1,
     0x1.812c8ea602e3cp-58},
    {0x1.2e025cab271d7p+0, -0x1.e857da2fa6000p-3, 0x1.b1ffff10d3d4dp-1,
     -0x1.efaddad27789cp-55},
    {0x1.2b404cf13cd82p+0, -0x1.cd3c8633d8000p-3, 0x1.b5fffce21165ap-1,
     0x1.3cb1719c61237p-58},
    {0x1.288b02c7ccb50p+0, -0x1.b26034c14a000p-3, 0x1.b9fffd950e674p-1,
     0x1.3f7d94194cep-56},
    {0x1.25e2263944de5p+0, -0x1.97c1c2f4fe000p-3, 0x1.be000139ca8afp-1,
     0x1.50ac4215d9bcp-56},
    {0x1.234563d8615b1p+0, -0x1.7d6023f800000p-3, 0x1.c20005b46df99p-1,
     0x1.beea653e9c1c9p-57},
    {0x1.20b46e33eaf38p+0, -0x1.633a71a05e000p-3, 0x1.c600040b9f7aep-1,
     -0x1.c079f274a70d6p-56},
    {0x1.1e2eefdcda3ddp+0, -0x1.494f5e9570000p-3, 0x1.ca0006255fd8ap-1,
     -0x1.a0b4076e84c1fp-56},
    {0x1.1bb4a580b3930p+0, -0x1.2f9e424e0a000p-3, 0x1.cdfffd94c095dp-1,
     0x1.8f933f99ab5d7p-55},
    {0x1.19453847f2200p+0, -0x1.162595afdc000p-3, 0x1.d1ffff975d6cfp-1,
     -0x1.82c08665fe1bep-58},
    {0x1.16e06c0d5d73cp+0, -0x1.f9c9a75bd8000p-4, 0x1.d5fffa2561c93p-1,
     -0x1.b04289bd295f3p-56},
    {0x1.1485f47b7e4c2p+0, -0x1.c7b575bf9c000p-4, 0x1.d9fff9d228b0cp-1,
     0x1.70251340fa236p-55},
    {0x1.12358ad0085d1p+0, -0x1.960c60ff48000p-4, 0x1.de00065bc7e16p-1,
     -0x1.5011e16a4d80cp-56},
    {0x1.0fef00f532227p+0, -0x1.64ce247b60000p-4, 0x1.e200002f64791p-1,
     0x1.9802f09ef62ep-55},
    {0x1.0db2077d03a8fp+0, -0x1.33f78b2014000p-4, 0x1.e600057d7a6d8p-1,
     -0x1.e0b75580cf7fap-56},
    {0x1.0b7e6d65980d9p+0, -0x1.0387d1a42c000p-4, 0x1.ea00027edc00cp-1,
     -0x1.c848309459811p-55},
    {0x1.0953efe7b408dp+0, -0x1.a6f9208b50000p-5, 0x1.ee0006cf5cb7cp-1,
     -0x1.f8027951576f4p-55},
    {0x1.07325cac53b83p+0, -0x1.47a954f770000p-5, 0x1.f2000782b7dccp-1,
     -0x1.f81d97274538fp-55},
    {0x1.05197e40d1b5cp+0, -0x1.d23a8c50c0000p-6, 0x1.f6000260c450ap-1,
     -0x1.071002727ffdcp-59},
    {0x1.03091c1208ea2p+0, -0x1.16a2629780000p-6, 0x1.f9fffe88cd533p-1,
     -0x1.81bdce1fda8bp-58},
    {0x1.0101025b37e21p+0, -0x1.720f8d8e80000p-8, 0x1.fdfffd50f8689p-1,
     0x1.7f91acb918e6ep-55},
    {0x1.fc07ef9caa76bp-1, 0x1.6fe53b1500000p-7, 0x1.0200004292367p+0,
     0x1.b7ff365324681p-54},
    {0x1.f4465d3f6f184p-1, 0x1.11ccce10f8000p-5, 0x1.05fffe3e3d668p+0,
     0x1.6fa08ddae957bp-55},
    {0x1.ecc079f84107fp-1, 0x1.c4dfc8c8b8000p-5, 0x1.0a0000a85a757p+0,
     -0x1.7e2de80d3fb91p-58},
    {0x1.e573a99975ae8p-1, 0x1.3aa321e574000p-4, 0x1.0e0001a5f3fccp+0,
     -0x1.1823305c5f014p-54},
    {0x1.de5d6f0bd3de6p-1, 0x1.918a0d08b8000p-4, 0x1.11ffff8afbaf5p+0,
     -0x1.bfabb6680bac2p-55},
    {0x1.d77b681ff38b3p-1, 0x1.e72e9da044000p-4, 0x1.15fffe54d91adp+0,
     -0x1.d7f121737e7efp-54},
    {0x1.d0cb5724de943p-1, 0x1.1dcd2507f6000p-3, 0x1.1a00011ac36e1p+0,
     0x1.c000a0516f5ffp-54},
    {0x1.ca4b2dc0e7563p-1, 0x1.476ab03dea000p-3, 0x1.1e00019c84248p+0,
     -0x1.082fbe4da5dap-54},
    {0x1.c3f8ee8d6cb51p-1, 0x1.7074377e22000p-3, 0x1.220000ffe5e6ep+0,
     -0x1.8fdd04c9cfb43p-55},
    {0x1.bdd2b4f020c4cp-1, 0x1.98ede8ba94000p-3, 0x1.26000269fd891p+0,
     0x1.cfe2a7994d182p-55},
    {0x1.b7d6c006015cap-1, 0x1.c0db86ad2e000p-3, 0x1.2a00029a6e6dap+0,
     -0x1.00273715e8bc5p-56},
    {0x1.b20366e2e338fp-1, 0x1.e840aafcee000p-3, 0x1.2dfffe0293e39p+0,
     0x1.b7c39dab2a6f9p-54},
    {0x1.ac57026295039p-1, 0x1.0790ab4678000p-2, 0x1.31ffff7dcf082p+0,
     0x1.df1336edc5254p-56},
    {0x1.a6d01bc2731ddp-1, 0x1.1ac056801c000p-2, 0x1.35ffff05a8b6p+0,
     -0x1.e03564ccd31ebp-54},
    {0x1.a16d3bc3ff18bp-1, 0x1.2db11d4fee000p-2, 0x1.3a0002e0eaeccp+0,
     0x1.5f0e74bd3a477p-56},
    {0x1.9c2d14967feadp-1, 0x1.406464ec58000p-2, 0x1.3e000043bb236p+0,
     0x1.c7dcb149d8833p-54},
    {0x1.970e4f47c9902p-1, 0x1.52dbe093af000p-2, 0x1.4200002d187ffp+0,
     0x1.e08afcf2d3d28p-56},
    {0x1.920fb3982bcf2p-1, 0x1.651902050d000p-2, 0x1.460000d387cb1p+0,
     0x1.20837856599a6p-55},
    {0x1.8d30187f759f1p-1, 0x1.771d2cdeaf000p-2, 0x1.4a00004569f89p+0,
     -0x1.9fa5c904fbcd2p-55},
    {0x1.886e5ebb9f66dp-1, 0x1.88e9c857d9000p-2, 0x1.4e000043543f3p+0,
     -0x1.81125ed175329p-56},
    {0x1.83c97b658b994p-1, 0x1.9a80155e16000p-2, 0x1.51fffcc027f0fp+0,
     0x1.883d8847754dcp-54},
    {0x1.7f405ffc61022p-1, 0x1.abe186ed3d000p-2, 0x1.55ffffd87b36fp+0,
     -0x1.709e731d02807p-55},
    {0x1.7ad22181415cap-1, 0x1.bd0f2aea0e000p-2, 0x1.59ffff21df7bap+0,
     0x1.7f79f68727b02p-55},
    {0x1.767dcf99eff8cp-1, 0x1.ce0a43dbf4000p-2, 0x1.5dfffebfc3481p+0,
     -0x1.180902e30e93ep-54},
};

static const double iree_math_log2_f64_polynomial[] = {
    -0x1.71547652b8339p-1, 0x1.ec709dc3a04bep-2,  -0x1.7154764702ffbp-2,
    0x1.2776c50034c48p-2,  -0x1.ec7b328ea92bcp-3, 0x1.a6225e117f92ep-3,
};

static const double iree_math_log2_f64_near_one[] = {
    -0x1.71547652b82fep-1, 0x1.ec709dc3a03f7p-2,  -0x1.71547652b7c3fp-2,
    0x1.2776c50f05be4p-2,  -0x1.ec709dd768fe5p-3, 0x1.a61761ec4e736p-3,
    -0x1.7153fbc64a79bp-3, 0x1.484d154f01b4ap-3,  -0x1.289e4a72c383cp-3,
    0x1.0b32f285aee66p-3,
};

typedef struct iree_math_log_f32_table_entry_t {
  // Reciprocal of the subinterval center.
  double inverse_center;
  // Natural logarithm of the subinterval center.
  double log_center;
} iree_math_log_f32_table_entry_t;

static const iree_math_log_f32_table_entry_t iree_math_log_f32_table[16] = {
    {0x1.661ec79f8f3bep+0, -0x1.57bf7808caadep-2},
    {0x1.571ed4aaf883dp+0, -0x1.2bef0a7c06ddbp-2},
    {0x1.49539f0f010bp+0, -0x1.01eae7f513a67p-2},
    {0x1.3c995b0b80385p+0, -0x1.b31d8a68224e9p-3},
    {0x1.30d190c8864a5p+0, -0x1.6574f0ac07758p-3},
    {0x1.25e227b0b8eap+0, -0x1.1aa2bc79c81p-3},
    {0x1.1bb4a4a1a343fp+0, -0x1.a4e76ce8c0e5ep-4},
    {0x1.12358f08ae5bap+0, -0x1.1973c5a611cccp-4},
    {0x1.0953f419900a7p+0, -0x1.252f438e10c1ep-5},
    {0x1p+0, 0x0p+0},
    {0x1.e608cfd9a47acp-1, 0x1.aa5aa5df25984p-5},
    {0x1.ca4b31f026aap-1, 0x1.c5e53aa362eb4p-4},
    {0x1.b2036576afce6p-1, 0x1.526e57720db08p-3},
    {0x1.9c2d163a1aa2dp-1, 0x1.bc2860d22477p-3},
    {0x1.886e6037841edp-1, 0x1.1058bc8a07ee1p-2},
    {0x1.767dcf5534862p-1, 0x1.4043057b6ee09p-2},
};

// These helpers receive only zeros, negative values, infinities, and NaNs.
static float iree_math_log_f32_special(uint32_t bits) {
  if ((bits & IREE_MATH_F32_MAGNITUDE_MASK) == 0) {
    return iree_math_f32_from_bits(UINT32_C(0xFF800000));
  }
  return bits == IREE_MATH_F32_INFINITY ? iree_math_f32_from_bits(bits)
                                        : iree_math_f32_canonical_nan();
}

static double iree_math_log_f64_special(uint64_t bits) {
  if ((bits & IREE_MATH_F64_MAGNITUDE_MASK) == 0) {
    return iree_math_f64_from_bits(UINT64_C(0xFFF0000000000000));
  }
  return bits == IREE_MATH_F64_INFINITY ? iree_math_f64_from_bits(bits)
                                        : iree_math_f64_canonical_nan();
}

// Normalize a positive finite input. The wrapped exponent remains signed in
// the encoded reduction domain, including for the smallest subnormal input.
static uint32_t iree_math_log_f32_normalize(float value, uint32_t bits) {
  if (IREE_UNLIKELY(bits < IREE_MATH_F32_MIN_NORMAL)) {
    return iree_math_f32_to_bits(value * 0x1p23f) - (UINT32_C(23) << 23);
  }
  return bits;
}

static uint64_t iree_math_log_f64_normalize(double value, uint64_t bits) {
  if (IREE_UNLIKELY(bits < IREE_MATH_F64_MIN_NORMAL)) {
    return iree_math_f64_to_bits(value * 0x1p52) - (UINT64_C(52) << 52);
  }
  return bits;
}

static int32_t iree_math_log_f64_exponent(uint64_t adjusted_bits) {
  const uint32_t encoded = (uint32_t)(adjusted_bits >> 52);
  return encoded >= 2048 ? (int32_t)encoded - 4096 : (int32_t)encoded;
}

// Retain the natural-log result in F64 until either the direct F32 rounding or
// the base-ten scale. Rounding logf before that scale would lose accuracy.
static double iree_math_log_f32_finite(uint32_t bits) {
  const uint32_t adjusted = bits - UINT32_C(0x3F330000);
  const uint32_t index = (adjusted >> 19) & 15;
  const uint32_t encoded = adjusted >> 23;
  const int32_t exponent =
      encoded >= 256 ? (int32_t)encoded - 512 : (int32_t)encoded;
  const double normalized =
      (double)iree_math_f32_from_bits(bits - (adjusted & UINT32_C(0xFF800000)));
  const iree_math_log_f32_table_entry_t* entry =
      &iree_math_log_f32_table[index];
  const double residual = normalized * entry->inverse_center - 1.0;
  const double base =
      entry->log_center + (double)exponent * 0x1.62e42fefa39efp-1;
  const double squared = residual * residual;
  double polynomial = 0x1.5575b0be00b6ap-2 * residual - 0x1.ffffef20a4123p-2;
  polynomial = -0x1.00ea348b88334p-2 * squared + polynomial;
  return polynomial * squared + (base + residual);
}

IREE_API_EXPORT float iree_math_log_f32(float value) {
  uint32_t bits = iree_math_f32_to_bits(value);
  if (IREE_UNLIKELY(bits - 1 >= IREE_MATH_F32_INFINITY - 1)) {
    return iree_math_log_f32_special(bits);
  }
  bits = iree_math_log_f32_normalize(value, bits);
  return (float)iree_math_log_f32_finite(bits);
}

IREE_API_EXPORT float iree_math_log2_f32(float value) {
  uint32_t bits = iree_math_f32_to_bits(value);
  if (IREE_UNLIKELY(bits - 1 >= IREE_MATH_F32_INFINITY - 1)) {
    return iree_math_log_f32_special(bits);
  }
  bits = iree_math_log_f32_normalize(value, bits);
  return iree_math_log2_f32_finite(bits);
}

IREE_API_EXPORT float iree_math_log10_f32(float value) {
  uint32_t bits = iree_math_f32_to_bits(value);
  if (IREE_UNLIKELY(bits - 1 >= IREE_MATH_F32_INFINITY - 1)) {
    return iree_math_log_f32_special(bits);
  }
  bits = iree_math_log_f32_normalize(value, bits);
  return (float)(iree_math_log_f32_finite(bits) * 0x1.bcb7b1526e50ep-2);
}

IREE_API_EXPORT double iree_math_log_f64(double value) {
  uint64_t bits = iree_math_f64_to_bits(value);
  // Near one, the small result needs a dedicated polynomial and compensated
  // quadratic term; table reduction alone cannot bound the relative error.
  const uint64_t lower = UINT64_C(0x3FEE000000000000);
  const uint64_t upper = UINT64_C(0x3FF1090000000000);
  if (IREE_UNLIKELY(bits - lower < upper - lower)) {
    const double* coefficients = iree_math_log_f64_near_one;
    const double residual = value - 1.0;
    const double squared = residual * residual;
    const double cubed = residual * squared;
    double result =
        cubed *
        (coefficients[1] + residual * coefficients[2] +
         squared * coefficients[3] +
         cubed *
             (coefficients[4] + residual * coefficients[5] +
              squared * coefficients[6] +
              cubed * (coefficients[7] + residual * coefficients[8] +
                       squared * coefficients[9] + cubed * coefficients[10])));
    const double shifted = residual * 0x1p27;
    const double residual_high = residual + shifted - shifted;
    const double residual_low = residual - residual_high;
    const double quadratic = residual_high * residual_high * coefficients[0];
    const double high = residual + quadratic;
    double low = residual - high + quadratic;
    low += coefficients[0] * residual_low * (residual_high + residual);
    result += low;
    return result + high;
  }
  if (IREE_UNLIKELY(bits - 1 >= IREE_MATH_F64_INFINITY - 1)) {
    return iree_math_log_f64_special(bits);
  }
  bits = iree_math_log_f64_normalize(value, bits);
  const uint64_t adjusted = bits - UINT64_C(0x3FE6000000000000);
  const uint32_t index = (uint32_t)(adjusted >> 45) & 127;
  const double exponent = (double)iree_math_log_f64_exponent(adjusted);
  const double normalized =
      iree_math_f64_from_bits(bits - (adjusted & UINT64_C(0xFFF0000000000000)));
  const iree_math_log_f64_table_entry_t* entry =
      &iree_math_log_f64_table[index];
  const double residual =
      (normalized - entry->center_high - entry->center_low) *
      entry->inverse_center;
  const double base = exponent * 0x1.62e42fefa3800p-1 + entry->log_center;
  const double high = base + residual;
  const double low = base - high + residual + exponent * 0x1.ef35793c76730p-45;
  const double squared = residual * residual;
  const double* coefficients = iree_math_log_f64_polynomial;
  return low + squared * coefficients[0] +
         residual * squared *
             (coefficients[1] + residual * coefficients[2] +
              squared * (coefficients[3] + residual * coefficients[4])) +
         high;
}

IREE_API_EXPORT double iree_math_log2_f64(double value) {
  const double inverse_ln2_high = 0x1.7154765200000p+0;
  const double inverse_ln2_low = 0x1.705fc2eefa200p-33;
  uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t lower = iree_math_f64_to_bits(1.0 - 0x1.5b51p-5);
  const uint64_t upper = iree_math_f64_to_bits(1.0 + 0x1.6ab2p-5);
  if (IREE_UNLIKELY(bits - lower < upper - lower)) {
    const double residual = value - 1.0;
    const double residual_high = iree_math_f64_from_bits(
        iree_math_f64_to_bits(residual) & UINT64_C(0xFFFFFFFF00000000));
    const double residual_low = residual - residual_high;
    const double high = residual_high * inverse_ln2_high;
    double low = residual_low * inverse_ln2_high + residual * inverse_ln2_low;
    const double squared = residual * residual;
    const double fourth = squared * squared;
    const double* coefficients = iree_math_log2_f64_near_one;
    const double quadratic =
        squared * (coefficients[0] + residual * coefficients[1]);
    double result = high + quadratic;
    low += high - result + quadratic;
    low +=
        fourth *
        (coefficients[2] + residual * coefficients[3] +
         squared * (coefficients[4] + residual * coefficients[5]) +
         fourth * (coefficients[6] + residual * coefficients[7] +
                   squared * (coefficients[8] + residual * coefficients[9])));
    return result + low;
  }
  if (IREE_UNLIKELY(bits - 1 >= IREE_MATH_F64_INFINITY - 1)) {
    return iree_math_log_f64_special(bits);
  }
  bits = iree_math_log_f64_normalize(value, bits);
  const uint64_t adjusted = bits - UINT64_C(0x3FE6000000000000);
  const uint32_t index = (uint32_t)(adjusted >> 46) & 63;
  const double exponent = (double)iree_math_log_f64_exponent(adjusted);
  const double normalized =
      iree_math_f64_from_bits(bits - (adjusted & UINT64_C(0xFFF0000000000000)));
  const iree_math_log_f64_table_entry_t* entry =
      &iree_math_log2_f64_table[index];
  const double residual =
      (normalized - entry->center_high - entry->center_low) *
      entry->inverse_center;
  const double residual_high = iree_math_f64_from_bits(
      iree_math_f64_to_bits(residual) & UINT64_C(0xFFFFFFFF00000000));
  const double residual_low = residual - residual_high;
  const double linear_high = residual_high * inverse_ln2_high;
  const double linear_low =
      residual_low * inverse_ln2_high + residual * inverse_ln2_low;
  const double base = exponent + entry->log_center;
  const double high = base + linear_high;
  const double low = base - high + linear_high + linear_low;
  const double squared = residual * residual;
  const double fourth = squared * squared;
  const double* coefficients = iree_math_log2_f64_polynomial;
  const double polynomial =
      coefficients[0] + residual * coefficients[1] +
      squared * (coefficients[2] + residual * coefficients[3]) +
      fourth * (coefficients[4] + residual * coefficients[5]);
  return low + squared * polynomial + high;
}

// Cancellation-sensitive logarithms adapted from OpenLibm revision
// 5fe399749f9276eaa0b8403e507470da05cbbb3f (e_log10.c) and musl revision
// 7813e9119eb5fe681b6750b8a033a0490bca0941 (log1p.c and log1pf.c).
//
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

static const float iree_math_log1p_f32_polynomial[4] = {
    0x1.5555540000000p-1f, 0x1.999c260000000p-2f, 0x1.23d3dc0000000p-2f,
    0x1.f13c4c0000000p-3f};

static const double iree_math_log1p_f64_polynomial[7] = {
    0x1.5555555555593p-1, 0x1.999999997fa04p-2, 0x1.2492494229359p-2,
    0x1.c71c51d8e78afp-3, 0x1.7466496cb03dep-3, 0x1.39a09d078c69fp-3,
    0x1.2f112df3e5244p-3};

IREE_API_EXPORT double iree_math_log10_f64(double value) {
  uint64_t bits = iree_math_f64_to_bits(value);
  if (IREE_UNLIKELY(bits - 1 >= IREE_MATH_F64_INFINITY - 1)) {
    return iree_math_log_f64_special(bits);
  }
  int32_t exponent = 0;
  if (IREE_UNLIKELY(bits < IREE_MATH_F64_MIN_NORMAL)) {
    value *= 0x1p54;
    bits = iree_math_f64_to_bits(value);
    exponent = -54;
  }
  uint32_t high_word = (uint32_t)(bits >> 32);
  exponent += (int32_t)(high_word >> 20) - 1023;
  high_word &= UINT32_C(0x000FFFFF);
  const uint32_t scale = (high_word + UINT32_C(0x95F64)) & UINT32_C(0x100000);
  value = iree_math_f64_from_bits(
      (bits & UINT64_C(0xFFFFFFFF)) |
      ((uint64_t)(high_word | (scale ^ UINT32_C(0x3FF00000))) << 32));
  exponent += (int32_t)(scale >> 20);
  const double residual = value - 1.0;
  const double quadratic = 0.5 * residual * residual;
  const double ratio = residual / (2.0 + residual);
  const double squared = ratio * ratio;
  const double fourth = squared * squared;
  const double* coefficients = iree_math_log1p_f64_polynomial;
  const double even =
      fourth *
      (coefficients[1] + fourth * (coefficients[3] + fourth * coefficients[5]));
  const double odd =
      squared *
      (coefficients[0] +
       fourth * (coefficients[2] +
                 fourth * (coefficients[4] + fourth * coefficients[6])));
  const double correction = ratio * (quadratic + (odd + even));
  const double high =
      iree_math_f64_from_bits(iree_math_f64_to_bits(residual - quadratic) &
                              UINT64_C(0xFFFFFFFF00000000));
  const double low = (residual - high) - quadratic + correction;
  const double inverse_ln10_high = 0x1.bcb7b15200000p-2;
  const double inverse_ln10_low = 0x1.b9438ca9aadd5p-36;
  const double log10_2_high = 0x1.34413509f6000p-2;
  const double log10_2_low = 0x1.9fef311f12b36p-42;
  double value_high = high * inverse_ln10_high;
  const double exponent_high = (double)exponent * log10_2_high;
  double value_low = (double)exponent * log10_2_low +
                     (low + high) * inverse_ln10_low + low * inverse_ln10_high;
  const double combined = exponent_high + value_high;
  value_low += (exponent_high - combined) + value_high;
  return value_low + combined;
}

IREE_API_EXPORT float iree_math_log1p_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F32_INFINITY)) {
    return bits == IREE_MATH_F32_INFINITY ? value
                                          : iree_math_f32_canonical_nan();
  }
  if (IREE_UNLIKELY(value <= -1.0f)) {
    return value == -1.0f ? iree_math_f32_from_bits(IREE_MATH_F32_SIGN_BIT |
                                                    IREE_MATH_F32_INFINITY)
                          : iree_math_f32_canonical_nan();
  }
  const uint32_t word = bits;
  if (IREE_UNLIKELY((word & UINT32_C(0x7FFFFFFF)) < UINT32_C(0x33800000))) {
    return value;
  }
  int32_t exponent = 0;
  float residual = value;
  float correction = 0.0f;
  // In this interval, 1+value needs no range reduction or rounding correction.
  const bool direct = (bits & IREE_MATH_F32_SIGN_BIT)
                          ? word <= UINT32_C(0xBE95F619)
                          : word < UINT32_C(0x3ED413D0);
  if (!direct) {
    const float sum = 1.0f + value;
    uint32_t normalized = iree_math_f32_to_bits(sum);
    normalized += UINT32_C(0x3F800000) - UINT32_C(0x3F3504F3);
    exponent = (int32_t)(normalized >> 23) - 127;
    // Recover the rounded sum's error while it can affect the logarithm.
    // The exponent bound also prevents an irrelevant underflow in correction.
    if (exponent < 25) {
      correction = exponent >= 2 ? 1.0f - (sum - value) : value - (sum - 1.0f);
      correction /= sum;
    }
    // Reduce 1+value to [sqrt(2)/2, sqrt(2)].
    normalized = (normalized & UINT32_C(0x007FFFFF)) + UINT32_C(0x3F3504F3);
    residual = iree_math_f32_from_bits(normalized) - 1.0f;
  }
  const float quadratic = 0.5f * residual * residual;
  const float ratio = residual / (2.0f + residual);
  const float squared = ratio * ratio;
  const float fourth = squared * squared;
  const float* coefficients = iree_math_log1p_f32_polynomial;
  const float even = fourth * (coefficients[1] + fourth * coefficients[3]);
  const float odd = squared * (coefficients[0] + fourth * coefficients[2]);
  const float scaled_exponent = (float)exponent;
  const float ln2_high = 0x1.62e300p-1f;
  const float ln2_low = 0x1.2fefa2p-17f;
  return ratio * (quadratic + (odd + even)) +
         (scaled_exponent * ln2_low + correction) - quadratic + residual +
         scaled_exponent * ln2_high;
}

IREE_API_EXPORT double iree_math_log1p_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F64_INFINITY)) {
    return bits == IREE_MATH_F64_INFINITY ? value
                                          : iree_math_f64_canonical_nan();
  }
  if (IREE_UNLIKELY(value <= -1.0)) {
    return value == -1.0 ? iree_math_f64_from_bits(IREE_MATH_F64_SIGN_BIT |
                                                   IREE_MATH_F64_INFINITY)
                         : iree_math_f64_canonical_nan();
  }
  const uint32_t word = (uint32_t)(bits >> 32);
  if (IREE_UNLIKELY((word & UINT32_C(0x7FFFFFFF)) < UINT32_C(0x3CA00000))) {
    return value;
  }
  int32_t exponent = 0;
  double residual = value;
  double correction = 0.0;
  // In this interval, 1+value needs no range reduction or rounding correction.
  const bool direct = (bits & IREE_MATH_F64_SIGN_BIT)
                          ? word <= UINT32_C(0xBFD2BEC4)
                          : word < UINT32_C(0x3FDA827A);
  if (!direct) {
    const double sum = 1.0 + value;
    uint32_t normalized = (uint32_t)(iree_math_f64_to_bits(sum) >> 32);
    normalized += UINT32_C(0x3FF00000) - UINT32_C(0x3FE6A09E);
    exponent = (int32_t)(normalized >> 20) - 1023;
    // Recover the rounded sum's error while it can affect the logarithm.
    // The exponent bound also prevents an irrelevant underflow in correction.
    if (exponent < 54) {
      correction = exponent >= 2 ? 1.0 - (sum - value) : value - (sum - 1.0);
      correction /= sum;
    }
    // Reduce 1+value to [sqrt(2)/2, sqrt(2)].
    normalized = (normalized & UINT32_C(0x000FFFFF)) + UINT32_C(0x3FE6A09E);
    residual = iree_math_f64_from_bits(
                   (iree_math_f64_to_bits(sum) & UINT64_C(0xFFFFFFFF)) |
                   ((uint64_t)normalized << 32)) -
               1.0;
  }
  const double quadratic = 0.5 * residual * residual;
  const double ratio = residual / (2.0 + residual);
  const double squared = ratio * ratio;
  const double fourth = squared * squared;
  const double* coefficients = iree_math_log1p_f64_polynomial;
  const double even =
      fourth *
      (coefficients[1] + fourth * (coefficients[3] + fourth * coefficients[5]));
  const double odd =
      squared *
      (coefficients[0] +
       fourth * (coefficients[2] +
                 fourth * (coefficients[4] + fourth * coefficients[6])));
  const double scaled_exponent = (double)exponent;
  const double ln2_high = 0x1.62e42fee00000p-1;
  const double ln2_low = 0x1.a39ef35793c76p-33;
  return ratio * (quadratic + (odd + even)) +
         (scaled_exponent * ln2_low + correction) - quadratic + residual +
         scaled_exponent * ln2_high;
}
