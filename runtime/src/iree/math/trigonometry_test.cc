// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/trigonometry.h"

#include "iree/math/elementary_test_util.h"

namespace iree::math::testing {
namespace {

TEST_F(ElementaryTest, TurnsF32ApproxExceptionalValues) {
  for (uint32_t input_bits : {UINT32_C(0x7F800000), UINT32_C(0xFF800000),
                              UINT32_C(0x7F800001), UINT32_C(0xFFC01234)}) {
    ExpectF32Mapping(iree_math_sin_turns_f32_approx, input_bits,
                     UINT32_C(0x7FC00000));
    ExpectF32Mapping(iree_math_cos_turns_f32_approx, input_bits,
                     UINT32_C(0x7FC00000));
  }
}

TEST_F(ElementaryTest, TurnsF32ApproxCardinalValues) {
  // Signed zero and full turns.
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x3F800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x3F800000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x3F800000));

  // Quarter and half turns.
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x3E800000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0xBE800000),
                   UINT32_C(0xBF800000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x3F000000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0xBF000000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x3E800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xBE800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x3F000000),
                   UINT32_C(0xBF800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xBF000000),
                   UINT32_C(0xBF800000));

  // Every f32 value with magnitude at least 2^23 is an integral turn.
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x4B000000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0xCB000000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x4B000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xCB000000),
                   UINT32_C(0x3F800000));
}

TEST_F(ElementaryTest, TurnsF32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x00000006));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0x80000006));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x3E000000),
                   UINT32_C(0x3F3504F3));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0x3E000000),
                   UINT32_C(0x3F3504F3));
  ExpectF32Mapping(iree_math_sin_turns_f32_approx, UINT32_C(0x465BFDF6),
                   UINT32_C(0x3D7B2B74));
  ExpectF32Mapping(iree_math_cos_turns_f32_approx, UINT32_C(0xC5CD59EC),
                   UINT32_C(0x3D7B2B74));
}

template <typename Bits>
void ExpectTrigonometricResult(Bits actual, Bits expected) {
  constexpr Bits sign = Bits{1} << (sizeof(Bits) * 8 - 1);
  constexpr Bits infinity = sizeof(Bits) == 4 ? 0x7F800000 : 0x7FF0000000000000;
  const Bits magnitude = expected & (sign - 1);
  if (magnitude == 0 || magnitude >= infinity) {
    EXPECT_EQ(actual, expected);
  } else {
    EXPECT_LE(actual > expected ? actual - expected : expected - actual,
              Bits{1});
  }
}

// MPFR 4.2.2 directed intervals, independently rounded to F32.
// Adjacent inputs cover tiny shortcuts, reduction boundaries, cancellation,
// poles, and the largest exponents. Odd/even symmetry exercises both signs.
TEST_F(ElementaryTest, RadianF32Boundaries) {
  const F32Function functions[] = {iree_math_sin_f32, iree_math_cos_f32,
                                   iree_math_tan_f32};
  const struct {
    // Positive input encoding, including exceptional payloads.
    uint32_t input;
    // Correctly rounded positive-input results in function order.
    uint32_t expected[3];
  } cases[] = {
      {0x00000000, {0x00000000, 0x3F800000, 0x00000000}},
      {0x00000001, {0x00000001, 0x3F800000, 0x00000001}},
      {0x007FFFFF, {0x007FFFFF, 0x3F800000, 0x007FFFFF}},
      {0x00800000, {0x00800000, 0x3F800000, 0x00800000}},
      {0x7F7FFFFF, {0xBF0599B3, 0x3F5A5F96, 0xBF1C9ECA}},
      {0x7F800000, {0x7FC00000, 0x7FC00000, 0x7FC00000}},
      {0x7F800001, {0x7FC00000, 0x7FC00000, 0x7FC00000}},
      {0x7FC00011, {0x7FC00000, 0x7FC00000, 0x7FC00000}},
      {0x397FFFFF, {0x397FFFFF, 0x3F800000, 0x397FFFFF}},
      {0x39800000, {0x39800000, 0x3F800000, 0x39800000}},
      {0x39800001, {0x39800001, 0x3F7FFFFF, 0x39800001}},
      {0x3F490FD9, {0x3F3504F2, 0x3F3504F4, 0x3F7FFFFD}},
      {0x3F490FDA, {0x3F3504F3, 0x3F3504F4, 0x3F7FFFFF}},
      {0x3F490FDB, {0x3F3504F3, 0x3F3504F3, 0x3F800000}},
      {0x4016CBE2, {0x3F3504F9, 0xBF3504EE, 0xBF800008}},
      {0x4016CBE3, {0x3F3504F6, 0xBF3504F0, 0xBF800004}},
      {0x4016CBE4, {0x3F3504F3, 0xBF3504F3, 0xBF800000}},
      {0x407B53D0, {0xBF3504F0, 0xBF3504F7, 0x3F7FFFF6}},
      {0x407B53D1, {0xBF3504F2, 0xBF3504F4, 0x3F7FFFFE}},
      {0x407B53D2, {0xBF3504F5, 0xBF3504F1, 0x3F800003}},
      {0x40AFEDDE, {0xBF3504FB, 0x3F3504EC, 0xBF80000A}},
      {0x40AFEDDF, {0xBF3504F5, 0x3F3504F1, 0xBF800002}},
      {0x40AFEDE0, {0xBF3504EF, 0x3F3504F7, 0xBF7FFFF5}},
      {0x40E231D4, {0x3F3504E8, 0x3F3504FE, 0x3F7FFFE1}},
      {0x40E231D5, {0x3F3504EE, 0x3F3504F9, 0x3F7FFFF1}},
      {0x40E231D6, {0x3F3504F3, 0x3F3504F3, 0x3F800000}},
      {0x42EFFFFF, {0x3F14A287, 0x3F506E74, 0x3F368E7A}},
      {0x42F00000, {0x3F14A2EF, 0x3F506E2A, 0x3F368F3B}},
      {0x42F00001, {0x3F14A357, 0x3F506DE0, 0x3F368FFC}},
      {0x3FC90FDA, {0x3F800000, 0x33A22169, 0x4B4A1BDA}},
      {0x3FC90FDB, {0x3F800000, 0xB33BBD2E, 0xCBAE8A4A}},
      {0x3FC90FDC, {0x3F800000, 0xB42EEF4C, 0xCABB50C8}},
      {0x40490FDA, {0x34222169, 0xBF800000, 0xB4222169}},
      {0x40490FDB, {0xB3BBBD2E, 0xBF800000, 0x33BBBD2E}},
      {0x40490FDC, {0xB4AEEF4C, 0xBF800000, 0x34AEEF4C}},
      {0x40C90FDA, {0xB4A22169, 0x3F800000, 0xB4A22169}},
      {0x40C90FDB, {0x343BBD2E, 0x3F800000, 0x343BBD2E}},
      {0x40C90FDC, {0x352EEF4C, 0x3F800000, 0x352EEF4C}},
      {0x41C90FDA, {0xB5A22169, 0x3F800000, 0xB5A22169}},
      {0x41C90FDB, {0x353BBD2E, 0x3F800000, 0x353BBD2E}},
      {0x41C90FDC, {0x362EEF4C, 0x3F800000, 0x362EEF4C}},
      {0x43C90FDA, {0xB7A22169, 0x3F800000, 0xB7A22169}},
      {0x43C90FDB, {0x373BBD2E, 0x3F800000, 0x373BBD2E}},
      {0x43C90FDC, {0x382EEF4C, 0x3F800000, 0x382EEF4C}},
      {0x49C90FDA, {0xBDA1F612, 0x3F7F32BF, 0xBDA27855}},
      {0x49C90FDB, {0x3D3BAC5B, 0x3F7FBB2C, 0x3D3BDEDE}},
      {0x49C90FDC, {0x3E2E15C9, 0x3F7C4601, 0x3E30A81A}},
      {0x72490FDA, {0xBE1B6D5E, 0x3F7D08AD, 0xBE1D3FC9}},
      {0x72490FDB, {0x3ECC418D, 0xBF6ABF13, 0xBEDEBFC7}},
      {0x72490FDC, {0xBF1EA3BF, 0x3F48EBEA, 0xBF4A209C}},
      {0x7F490FDA, {0x3F21A943, 0x3F467F9F, 0x3F507DD6}},
      {0x7F490FDB, {0x3F089EFB, 0xBF587F15, 0xBF218CC8}},
      {0x7F490FDC, {0xBF78C000, 0xBE71F91A, 0x408395BC}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message() << "input=" << std::hex << test.input);
    for (size_t operation = 0; operation < 3; ++operation) {
      SCOPED_TRACE(operation);
      for (uint32_t sign : {UINT32_C(0), UINT32_C(0x80000000)}) {
        const uint32_t expected = test.expected[operation];
        const bool odd = operation != 1;
        const uint32_t signed_expected =
            odd && (expected & UINT32_C(0x7FFFFFFF)) <= UINT32_C(0x7F800000)
                ? expected ^ sign
                : expected;
        const uint32_t actual =
            F32ToBits(functions[operation](F32FromBits(test.input | sign)));
        ExpectTrigonometricResult(actual, signed_expected);
      }
    }
  }
}

// MPFR 4.2.2 directed intervals, independently rounded to F64.
// Adjacent inputs cover tiny shortcuts, reduction boundaries, cancellation,
// poles, and the largest exponents. Odd/even symmetry exercises both signs.
TEST_F(ElementaryTest, RadianF64Boundaries) {
  const F64Function functions[] = {iree_math_sin_f64, iree_math_cos_f64,
                                   iree_math_tan_f64};
  const struct {
    // Positive input encoding, including exceptional payloads.
    uint64_t input;
    // Correctly rounded positive-input results in function order.
    uint64_t expected[3];
  } cases[] = {
      {0x0000000000000000,
       {0x0000000000000000, 0x3FF0000000000000, 0x0000000000000000}},
      {0x0000000000000001,
       {0x0000000000000001, 0x3FF0000000000000, 0x0000000000000001}},
      {0x000FFFFFFFFFFFFF,
       {0x000FFFFFFFFFFFFF, 0x3FF0000000000000, 0x000FFFFFFFFFFFFF}},
      {0x0010000000000000,
       {0x0010000000000000, 0x3FF0000000000000, 0x0010000000000000}},
      {0x7FEFFFFFFFFFFFFF,
       {0x3F7452FC98B34E97, 0xBFEFFFE62ECFAB75, 0xBF74530CFE729484}},
      {0x7FF0000000000000,
       {0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000}},
      {0x7FF0000000000001,
       {0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000}},
      {0x7FF8000000000011,
       {0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000}},
      {0x3E3FFFFFFFFFFFFF,
       {0x3E3FFFFFFFFFFFFF, 0x3FF0000000000000, 0x3E3FFFFFFFFFFFFF}},
      {0x3E40000000000000,
       {0x3E40000000000000, 0x3FF0000000000000, 0x3E40000000000000}},
      {0x3E40000000000001,
       {0x3E40000000000001, 0x3FF0000000000000, 0x3E40000000000001}},
      {0x3E46A09DFFFFFFFF,
       {0x3E46A09DFFFFFFFF, 0x3FF0000000000000, 0x3E46A09DFFFFFFFF}},
      {0x3E46A09E00000000,
       {0x3E46A09E00000000, 0x3FF0000000000000, 0x3E46A09E00000000}},
      {0x3E46A09E00000001,
       {0x3E46A09E00000001, 0x3FF0000000000000, 0x3E46A09E00000001}},
      {0x3E4FFFFFFFFFFFFF,
       {0x3E4FFFFFFFFFFFFF, 0x3FEFFFFFFFFFFFFF, 0x3E50000000000000}},
      {0x3E50000000000000,
       {0x3E50000000000000, 0x3FEFFFFFFFFFFFFF, 0x3E50000000000000}},
      {0x3E50000000000001,
       {0x3E50000000000001, 0x3FEFFFFFFFFFFFFF, 0x3E50000000000001}},
      {0x3FE59427FFFFFFFF,
       {0x3FE3FAE8600608F6, 0x3FE8FEF3A3AA8B7F, 0x3FE99427887A14D4}},
      {0x3FE5942800000000,
       {0x3FE3FAE8600608F7, 0x3FE8FEF3A3AA8B7E, 0x3FE99427887A14D6}},
      {0x3FE5942800000001,
       {0x3FE3FAE8600608F8, 0x3FE8FEF3A3AA8B7E, 0x3FE99427887A14D8}},
      {0x3FE921FBFFFFFFFF,
       {0x3FE6A09EDFEE58B9, 0x3FE6A09DED101C54, 0x3FF00000ABBBD680}},
      {0x3FE921FC00000000,
       {0x3FE6A09EDFEE58BA, 0x3FE6A09DED101C53, 0x3FF00000ABBBD681}},
      {0x3FE921FC00000001,
       {0x3FE6A09EDFEE58BB, 0x3FE6A09DED101C53, 0x3FF00000ABBBD682}},
      {0x4002D97CFFFFFFFF,
       {0x3FE6A09CFA31D5C1, 0xBFE6A09FD2CC8AEF, 0xBFEFFFFBF9994F67}},
      {0x4002D97D00000000,
       {0x3FE6A09CFA31D5BE, 0xBFE6A09FD2CC8AF2, 0xBFEFFFFBF9994F5F}},
      {0x4002D97D00000001,
       {0x3FE6A09CFA31D5BB, 0xBFE6A09FD2CC8AF5, 0xBFEFFFFBF9994F57}},
      {0x400F6A7AFFFFFFFF,
       {0xBFE6A0A0C5AAB2F8, 0xBFE6A09C075384FD, 0x3FF000035AAB7884}},
      {0x400F6A7B00000000,
       {0xBFE6A0A0C5AAB2FB, 0xBFE6A09C075384FA, 0x3FF000035AAB7888}},
      {0x400F6A7B00000001,
       {0xBFE6A0A0C5AAB2FE, 0xBFE6A09C075384F7, 0x3FF000035AAB788C}},
      {0x4015FDBCFFFFFFFF,
       {0xBFE6A0984060C5BA, 0x3FE6A0A48C9C061D, 0xBFEFFFEE9BBF3144}},
      {0x4015FDBD00000000,
       {0xBFE6A0984060C5B5, 0x3FE6A0A48C9C0622, 0xBFEFFFEE9BBF3134}},
      {0x4015FDBD00000001,
       {0xBFE6A0984060C5AF, 0x3FE6A0A48C9C0628, 0xBFEFFFEE9BBF3124}},
      {0x401C463BFFFFFFFF,
       {0x3FE6A0A57F79FB6C, 0x3FE6A0974D82423D, 0x3FF0000A099D9022}},
      {0x401C463C00000000,
       {0x3FE6A0A57F79FB72, 0x3FE6A0974D824237, 0x3FF0000A099D902A}},
      {0x401C463C00000001,
       {0x3FE6A0A57F79FB78, 0x3FE6A0974D824232, 0x3FF0000A099D9032}},
      {0x413921FAFFFFFFFF,
       {0xBFD4B02E5C25AF00, 0x3FEE483125700AC3, 0xBFD5DCA6BD5E8435}},
      {0x413921FB00000000,
       {0xBFD4B02E5BE91E9E, 0x3FEE4831257A62DA, 0xBFD5DCA6BD170C6F}},
      {0x413921FB00000001,
       {0xBFD4B02E5BAC8E3C, 0x3FEE48312584BAF1, 0xBFD5DCA6BCCF94A8}},
      {0x3FF921FB54442D17,
       {0x3FF0000000000000, 0x3CB469898CC51702, 0x4329153D9443ED0B}},
      {0x3FF921FB54442D18,
       {0x3FF0000000000000, 0x3C91A62633145C07, 0x434D02967C31CDB5}},
      {0x3FF921FB54442D19,
       {0x3FF0000000000000, 0xBCA72CECE675D1FD, 0xC33617A15494767A}},
      {0x400921FB54442D17,
       {0x3CC469898CC51702, 0xBFF0000000000000, 0xBCC469898CC51702}},
      {0x400921FB54442D18,
       {0x3CA1A62633145C07, 0xBFF0000000000000, 0xBCA1A62633145C07}},
      {0x400921FB54442D19,
       {0xBCB72CECE675D1FD, 0xBFF0000000000000, 0x3CB72CECE675D1FD}},
      {0x401921FB54442D17,
       {0xBCD469898CC51702, 0x3FF0000000000000, 0xBCD469898CC51702}},
      {0x401921FB54442D18,
       {0xBCB1A62633145C07, 0x3FF0000000000000, 0xBCB1A62633145C07}},
      {0x401921FB54442D19,
       {0x3CC72CECE675D1FD, 0x3FF0000000000000, 0x3CC72CECE675D1FD}},
      {0x403921FB54442D17,
       {0xBCF469898CC51702, 0x3FF0000000000000, 0xBCF469898CC51702}},
      {0x403921FB54442D18,
       {0xBCD1A62633145C07, 0x3FF0000000000000, 0xBCD1A62633145C07}},
      {0x403921FB54442D19,
       {0x3CE72CECE675D1FD, 0x3FF0000000000000, 0x3CE72CECE675D1FD}},
      {0x407921FB54442D17,
       {0xBD3469898CC51702, 0x3FF0000000000000, 0xBD3469898CC51702}},
      {0x407921FB54442D18,
       {0xBD11A62633145C07, 0x3FF0000000000000, 0xBD11A62633145C07}},
      {0x407921FB54442D19,
       {0x3D272CECE675D1FD, 0x3FF0000000000000, 0x3D272CECE675D1FD}},
      {0x413921FB54442D17,
       {0xBDF469898CC51702, 0x3FF0000000000000, 0xBDF469898CC51702}},
      {0x413921FB54442D18,
       {0xBDD1A62633145C07, 0x3FF0000000000000, 0xBDD1A62633145C07}},
      {0x413921FB54442D19,
       {0x3DE72CECE675D1FD, 0x3FF0000000000000, 0x3DE72CECE675D1FD}},
      {0x414921FB54442D17,
       {0xBE0469898CC51702, 0x3FF0000000000000, 0xBE0469898CC51702}},
      {0x414921FB54442D18,
       {0xBDE1A62633145C07, 0x3FF0000000000000, 0xBDE1A62633145C07}},
      {0x414921FB54442D19,
       {0x3DF72CECE675D1FD, 0x3FF0000000000000, 0x3DF72CECE675D1FD}},
      {0x432921FB54442D17,
       {0xBFE30E4DB8C01C25, 0x3FE9B51C56861C02, 0xBFE7B8638A8D46EB}},
      {0x432921FB54442D18,
       {0xBFC197D8B5F1B887, 0x3FEFB240562F5A4D, 0xBFC1C30050DFDF16}},
      {0x432921FB54442D19,
       {0x3FD6AC1C2AC7A76D, 0x3FEDECBD7EF6CDCD, 0x3FD83E9D9488CA8B}},
      {0x5F4921FB54442D17,
       {0xBFE3B537D4F52F5B, 0xBFE9360A5C4250A7, 0x3FE903CE79E18A6F}},
      {0x5F4921FB54442D18,
       {0x3FD7743240B6335D, 0x3FEDC6174990A7C7, 0x3FD935225CA489F9}},
      {0x5F4921FB54442D19,
       {0xBFB639A753AA8138, 0xBFEFE111B139ACE8, 0x3FB64F37BC04A7EA}},
      {0x7FD921FB54442D17,
       {0xBFC4C8A62ABA4765, 0xBFEF93498771F3A1, 0x3FC51034F90A5974}},
      {0x7FD921FB54442D18,
       {0xBFED0D611BC945D1, 0xBFDAD48D63FC690E, 0x4001533705A0FC7F}},
      {0x7FD921FB54442D19,
       {0xBFEB67B4933D5810, 0x3FE085822CF2786B, 0xBFFA8A3F75E7465D}},
      {0x7506AC5B262CA1FE,
       {0xBFE392680B64726C, 0x3FE9511A46CE5CDD, 0xBFE8BD10D78F5F84}},
      {0x7506AC5B262CA1FF,
       {0x3FF0000000000000, 0xBC214AE72E6BA22F, 0xC3BD9BA9A7975636}},
      {0x7506AC5B262CA200,
       {0xBFE392680B64726C, 0xBFE9511A46CE5CDD, 0x3FE8BD10D78F5F84}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message() << "input=" << std::hex << test.input);
    for (size_t operation = 0; operation < 3; ++operation) {
      SCOPED_TRACE(operation);
      for (uint64_t sign : {UINT64_C(0), UINT64_C(0x8000000000000000)}) {
        const uint64_t expected = test.expected[operation];
        const bool odd = operation != 1;
        const uint64_t signed_expected =
            odd && (expected & UINT64_C(0x7FFFFFFFFFFFFFFF)) <=
                        UINT64_C(0x7FF0000000000000)
                ? expected ^ sign
                : expected;
        const uint64_t actual =
            F64ToBits(functions[operation](F64FromBits(test.input | sign)));
        ExpectTrigonometricResult(actual, signed_expected);
      }
    }
  }
}

// MPFR 4.2.2 directed intervals, independently rounded to F64.
// Adjacent inputs cover tiny values, quadrant transitions, parity limits, and
// cardinal values. Odd/even symmetry exercises both signs.
TEST_F(ElementaryTest, TurnsF64Boundaries) {
  const F64Function functions[] = {iree_math_sin_turns_f64,
                                   iree_math_cos_turns_f64};
  const struct {
    // Positive input encoding, including exceptional payloads.
    uint64_t input;
    // Correctly rounded positive-input results in function order.
    uint64_t expected[2];
  } cases[] = {
      {0x0000000000000000, {0x0000000000000000, 0x3FF0000000000000}},
      {0x0000000000000001, {0x0000000000000006, 0x3FF0000000000000}},
      {0x000FFFFFFFFFFFFF, {0x003921FB54442D17, 0x3FF0000000000000}},
      {0x0010000000000000, {0x003921FB54442D18, 0x3FF0000000000000}},
      {0x7FEFFFFFFFFFFFFF, {0x0000000000000000, 0x3FF0000000000000}},
      {0x7FF0000000000000, {0x7FF8000000000000, 0x7FF8000000000000}},
      {0x7FF0000000000001, {0x7FF8000000000000, 0x7FF8000000000000}},
      {0x7FF8000000000011, {0x7FF8000000000000, 0x7FF8000000000000}},
      {0x3E0FFFFFFFFFFFFF, {0x3E3921FB54442D17, 0x3FF0000000000000}},
      {0x3E10000000000000, {0x3E3921FB54442D18, 0x3FF0000000000000}},
      {0x3E10000000000001, {0x3E3921FB54442D1A, 0x3FF0000000000000}},
      {0x3FBFFFFFFFFFFFFF, {0x3FE6A09E667F3BCC, 0x3FE6A09E667F3BCD}},
      {0x3FC0000000000000, {0x3FE6A09E667F3BCD, 0x3FE6A09E667F3BCD}},
      {0x3FC0000000000001, {0x3FE6A09E667F3BCE, 0x3FE6A09E667F3BCB}},
      {0x3FCFFFFFFFFFFFFF, {0x3FF0000000000000, 0x3CA921FB54442D18}},
      {0x3FD0000000000000, {0x3FF0000000000000, 0x0000000000000000}},
      {0x3FD0000000000001, {0x3FF0000000000000, 0xBCB921FB54442D18}},
      {0x3FD7FFFFFFFFFFFF, {0x3FE6A09E667F3BCF, 0xBFE6A09E667F3BCA}},
      {0x3FD8000000000000, {0x3FE6A09E667F3BCD, 0xBFE6A09E667F3BCD}},
      {0x3FD8000000000001, {0x3FE6A09E667F3BCA, 0xBFE6A09E667F3BCF}},
      {0x3FDFFFFFFFFFFFFF, {0x3CB921FB54442D18, 0xBFF0000000000000}},
      {0x3FE0000000000000, {0x0000000000000000, 0xBFF0000000000000}},
      {0x3FE0000000000001, {0xBCC921FB54442D18, 0xBFF0000000000000}},
      {0x3FE3FFFFFFFFFFFF, {0xBFE6A09E667F3BC8, 0xBFE6A09E667F3BD1}},
      {0x3FE4000000000000, {0xBFE6A09E667F3BCD, 0xBFE6A09E667F3BCD}},
      {0x3FE4000000000001, {0xBFE6A09E667F3BD1, 0xBFE6A09E667F3BC8}},
      {0x3FE7FFFFFFFFFFFF, {0xBFF0000000000000, 0xBCC921FB54442D18}},
      {0x3FE8000000000000, {0xBFF0000000000000, 0x0000000000000000}},
      {0x3FE8000000000001, {0xBFF0000000000000, 0x3CC921FB54442D18}},
      {0x3FEBFFFFFFFFFFFF, {0xBFE6A09E667F3BD1, 0x3FE6A09E667F3BC8}},
      {0x3FEC000000000000, {0xBFE6A09E667F3BCD, 0x3FE6A09E667F3BCD}},
      {0x3FEC000000000001, {0xBFE6A09E667F3BC8, 0x3FE6A09E667F3BD1}},
      {0x3FEFFFFFFFFFFFFF, {0xBCC921FB54442D18, 0x3FF0000000000000}},
      {0x3FF0000000000000, {0x0000000000000000, 0x3FF0000000000000}},
      {0x3FF0000000000001, {0x3CD921FB54442D18, 0x3FF0000000000000}},
      {0x3FF3FFFFFFFFFFFF, {0x3FF0000000000000, 0x3CD921FB54442D18}},
      {0x3FF4000000000000, {0x3FF0000000000000, 0x0000000000000000}},
      {0x3FF4000000000001, {0x3FF0000000000000, 0xBCD921FB54442D18}},
      {0x413000003FFFFFFF, {0x3FF0000000000000, 0x3E1921FB54442D18}},
      {0x4130000040000000, {0x3FF0000000000000, 0x0000000000000000}},
      {0x4130000040000001, {0x3FF0000000000000, 0xBE1921FB54442D18}},
      {0x41D00000000FFFFF, {0x3FEFFFFFFFFFD886, 0x3EB921FB544422C2}},
      {0x41D0000000100000, {0x3FF0000000000000, 0x0000000000000000}},
      {0x41D0000000100001, {0x3FEFFFFFFFFFD886, 0xBEB921FB544422C2}},
      {0x430FFFFFFFFFFFFF, {0xBFE6A09E667F3BCD, 0x3FE6A09E667F3BCD}},
      {0x4310000000000000, {0x0000000000000000, 0x3FF0000000000000}},
      {0x4310000000000001, {0x3FF0000000000000, 0x0000000000000000}},
      {0x431FFFFFFFFFFFFF, {0xBFF0000000000000, 0x0000000000000000}},
      {0x4320000000000000, {0x0000000000000000, 0x3FF0000000000000}},
      {0x4320000000000001, {0x0000000000000000, 0xBFF0000000000000}},
      {0x432FFFFFFFFFFFFF, {0x0000000000000000, 0xBFF0000000000000}},
      {0x4330000000000000, {0x0000000000000000, 0x3FF0000000000000}},
      {0x4330000000000001, {0x0000000000000000, 0x3FF0000000000000}},
      {0x433FFFFFFFFFFFFF, {0x0000000000000000, 0x3FF0000000000000}},
      {0x4340000000000000, {0x0000000000000000, 0x3FF0000000000000}},
      {0x4340000000000001, {0x0000000000000000, 0x3FF0000000000000}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message() << "input=" << std::hex << test.input);
    for (size_t operation = 0; operation < 2; ++operation) {
      SCOPED_TRACE(operation);
      for (uint64_t sign : {UINT64_C(0), UINT64_C(0x8000000000000000)}) {
        const uint64_t expected = test.expected[operation];
        const bool odd = operation != 1;
        const uint64_t signed_expected =
            odd && (expected & UINT64_C(0x7FFFFFFFFFFFFFFF)) <=
                        UINT64_C(0x7FF0000000000000)
                ? expected ^ sign
                : expected;
        const uint64_t actual =
            F64ToBits(functions[operation](F64FromBits(test.input | sign)));
        // Structural cardinals require exact unit values, not an ULP bound.
        if ((expected & UINT64_C(0x7FFFFFFFFFFFFFFF)) ==
            UINT64_C(0x3FF0000000000000)) {
          EXPECT_EQ(actual, signed_expected);
        } else {
          ExpectTrigonometricResult(actual, signed_expected);
        }
      }
    }
  }
}

}  // namespace
}  // namespace iree::math::testing
