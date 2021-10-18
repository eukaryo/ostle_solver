#include<iostream>
#include<iomanip>
#include<chrono>
#include<fstream>

#include<vector>
#include<string>
#include<map>
#include<set>
#include<unordered_map>
#include<unordered_set>

#include<algorithm>
#include<array>
#include<bitset>
#include<cassert>
#include<cstdint>
#include<exception>
#include<functional>
#include<limits>
#include<queue>
#include<regex>
#include<random>

//#include <omp.h>

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

alignas(32) const static uint8_t transpose_5x5_table[32] = { 0,5,10,15,20,1,6,11,16,21,2,7,12,17,22,3,8,13,18,23,4,9,14,19,24,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

alignas(32) const static uint8_t vertical_mirror_5x5_table[32] = { 20,21,22,23,24,15,16,17,18,19,10,11,12,13,14,5,6,7,8,9,0,1,2,3,4,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

alignas(32) const static uint8_t horizontal_mirror_5x5_table[32] = { 4,3,2,1,0,9,8,7,6,5,14,13,12,11,10,19,18,17,16,15,24,23,22,21,20,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

uint64_t code_unique1(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	const __m256i X2_mask_1_lo_horizontal = _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal)));

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	_mm256_storeu_si256((__m256i*)result1, tt);
	_mm256_storeu_si256((__m256i*)result2, zz);

	for (int i = 0; i < 4; ++i) {
		result1[i] = std::min(result1[i], result2[i]);
	}
	for (int i = 1; i < 4; ++i) {
		result1[0] = std::min(result1[0], result1[i]);
	}
	return result1[0];

	//const bool b00 = result1[0] < result2[0];
	//const uint64_t r0 = b00 ? result1[0] : result2[0];
	//const bool b01 = result1[1] < result2[1];
	//const uint64_t r1 = b01 ? result1[1] : result2[1];
	//const bool b02 = result1[2] < result2[2];
	//const uint64_t r2 = b02 ? result1[2] : result2[2];
	//const bool b03 = result1[3] < result2[3];
	//const uint64_t r3 = b03 ? result1[3] : result2[3];
	//const bool b10 = r0 < r1;
	//const uint64_t r4 = b10 ? r0 : r1;
	//const bool b11 = r2 < r3;
	//const uint64_t r5 = b11 ? r2 : r3;
	//const bool b20 = r4 < r5;
	//return b20 ? r4 : r5;
}

uint64_t code_unique2(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	const __m256i X2_mask_1_lo_horizontal = _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal)));

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	_mm256_storeu_si256((__m256i*)result1, tt);
	_mm256_storeu_si256((__m256i*)result2, zz);

	//for (int i = 0; i < 4; ++i) {
	//	result1[i] = std::min(result1[i], result2[i]);
	//}
	//for (int i = 1; i < 4; ++i) {
	//	result1[0] = std::min(result1[0], result1[i]);
	//}
	//return result1[0];

	const bool b00 = result1[0] < result2[0];
	const uint64_t r0 = b00 ? result1[0] : result2[0];
	const bool b01 = result1[1] < result2[1];
	const uint64_t r1 = b01 ? result1[1] : result2[1];
	const bool b02 = result1[2] < result2[2];
	const uint64_t r2 = b02 ? result1[2] : result2[2];
	const bool b03 = result1[3] < result2[3];
	const uint64_t r3 = b03 ? result1[3] : result2[3];
	const bool b10 = r0 < r1;
	const uint64_t r4 = b10 ? r0 : r1;
	const bool b11 = r2 < r3;
	const uint64_t r5 = b11 ? r2 : r3;
	const bool b20 = r4 < r5;
	return b20 ? r4 : r5;
}

uint64_t code_unique3(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	const __m256i X2_mask_1_lo_horizontal = _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal)));

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	const __m256i a1 = _mm256_sub_epi64(zz, tt);
	const __m256i a2 = _mm256_srai_epi32(a1, 32);
	const __m256i a3 = _mm256_shuffle_epi32(a2, 0b11110101);
	const __m256i a4 = _mm256_blendv_epi8(tt, zz, a3);
	alignas(32) uint64_t result[4] = {};
	_mm256_storeu_si256((__m256i*)result, a4);

	result[0] = std::min(result[0], result[1]);
	result[0] = std::min(result[0], result[2]);
	result[0] = std::min(result[0], result[3]);
	return result[0];

	//alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	//_mm256_storeu_si256((__m256i*)result1, tt);
	//_mm256_storeu_si256((__m256i*)result2, zz);

	//for (int i = 0; i < 4; ++i) {
	//	result1[i] = std::min(result1[i], result2[i]);
	//}
	//for (int i = 1; i < 4; ++i) {
	//	result1[0] = std::min(result1[0], result1[i]);
	//}
	//return result1[0];

	//const bool b00 = result1[0] < result2[0];
	//const uint64_t r0 = b00 ? result1[0] : result2[0];
	//const bool b01 = result1[1] < result2[1];
	//const uint64_t r1 = b01 ? result1[1] : result2[1];
	//const bool b02 = result1[2] < result2[2];
	//const uint64_t r2 = b02 ? result1[2] : result2[2];
	//const bool b03 = result1[3] < result2[3];
	//const uint64_t r3 = b03 ? result1[3] : result2[3];
	//const bool b10 = r0 < r1;
	//const uint64_t r4 = b10 ? r0 : r1;
	//const bool b11 = r2 < r3;
	//const uint64_t r5 = b11 ? r2 : r3;
	//const bool b20 = r4 < r5;
	//return b20 ? r4 : r5;
}

uint64_t code_unique4(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	const __m256i X2_mask_1_lo_horizontal = _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal)));

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	const __m256i a1 = _mm256_sub_epi64(zz, tt);
	const __m256i a2 = _mm256_srai_epi32(a1, 32);
	const __m256i a3 = _mm256_shuffle_epi32(a2, 0b11110101);
	const __m256i a4 = _mm256_blendv_epi8(tt, zz, a3);
	alignas(32) uint64_t result[4] = {};
	_mm256_storeu_si256((__m256i*)result, a4);

	result[0] = std::min(result[0], result[1]);
	result[2] = std::min(result[2], result[3]);
	result[0] = std::min(result[0], result[2]);
	return result[0];

	//alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	//_mm256_storeu_si256((__m256i*)result1, tt);
	//_mm256_storeu_si256((__m256i*)result2, zz);

	//for (int i = 0; i < 4; ++i) {
	//	result1[i] = std::min(result1[i], result2[i]);
	//}
	//for (int i = 1; i < 4; ++i) {
	//	result1[0] = std::min(result1[0], result1[i]);
	//}
	//return result1[0];

	//const bool b00 = result1[0] < result2[0];
	//const uint64_t r0 = b00 ? result1[0] : result2[0];
	//const bool b01 = result1[1] < result2[1];
	//const uint64_t r1 = b01 ? result1[1] : result2[1];
	//const bool b02 = result1[2] < result2[2];
	//const uint64_t r2 = b02 ? result1[2] : result2[2];
	//const bool b03 = result1[3] < result2[3];
	//const uint64_t r3 = b03 ? result1[3] : result2[3];
	//const bool b10 = r0 < r1;
	//const uint64_t r4 = b10 ? r0 : r1;
	//const bool b11 = r2 < r3;
	//const uint64_t r5 = b11 ? r2 : r3;
	//const bool b20 = r4 < r5;
	//return b20 ? r4 : r5;
}
uint64_t code_unique5(const uint64_t code) {

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	constexpr uint64_t mask_1_lo_vertical = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi_vertical = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo_vertical = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi_vertical = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo_vertical = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi_vertical = 0b11111'11111'11111'11111'00000ULL;

	uint64_t b = code, r;
	r = ((b >> 5) & X2(mask_1_lo_vertical)) | ((b << 5) & X2(mask_1_hi_vertical));
	r = ((r >> 10) & X2(mask_2_lo_vertical)) | ((r << 10) & X2(mask_2_hi_vertical));
	r = ((b >> 20) & X2(mask_3_lo_vertical)) | ((r << 5) & X2(mask_3_hi_vertical));

	constexpr uint64_t mask_1_lo_horizontal = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi_horizontal = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo_horizontal = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi_horizontal = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo_horizontal = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi_horizontal = 0b11110'11110'11110'11110'11110ULL;

	constexpr int64_t FULL = ((1LL << 50) - 1LL) << 5;

	const __m256i X2_mask_1_lo_horizontal = _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal)));

	__m256i bb0 = _mm256_set_epi64x(int64_t(code), int64_t(r), int64_t(code), int64_t(r));

	const __m256i tt1lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_lo_horizontal)), int64_t(X2(mask_1_lo_horizontal))));
	const __m256i tt1hi = _mm256_and_si256(_mm256_sllv_epi64(bb0, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_1_hi_horizontal)), int64_t(X2(mask_1_hi_horizontal))));
	const __m256i tt1 = _mm256_or_si256(tt1lo, tt1hi);

	const __m256i tt2lo = _mm256_and_si256(_mm256_srlv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_lo_horizontal)), int64_t(X2(mask_2_lo_horizontal))));
	const __m256i tt2hi = _mm256_and_si256(_mm256_sllv_epi64(tt1, _mm256_set_epi64x(0, 0, 2, 2)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_2_hi_horizontal)), int64_t(X2(mask_2_hi_horizontal))));
	const __m256i tt2 = _mm256_or_si256(tt2lo, tt2hi);

	const __m256i tt3lo = _mm256_and_si256(_mm256_srlv_epi64(bb0, _mm256_set_epi64x(0, 0, 4, 4)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_lo_horizontal)), int64_t(X2(mask_3_lo_horizontal))));
	const __m256i tt3hi = _mm256_and_si256(_mm256_sllv_epi64(tt2, _mm256_set_epi64x(0, 0, 1, 1)), _mm256_set_epi64x(FULL, FULL, int64_t(X2(mask_3_hi_horizontal)), int64_t(X2(mask_3_hi_horizontal))));
	const __m256i tt3 = _mm256_or_si256(tt3lo, tt3hi);

	const int64_t pos0 = int64_t(b % 32);
	const int64_t pos1 = int64_t(vertical_mirror_5x5_table[b % 32]);
	const int64_t pos2 = int64_t(horizontal_mirror_5x5_table[b % 32]);
	const int64_t pos3 = int64_t(horizontal_mirror_5x5_table[vertical_mirror_5x5_table[b % 32]]);

	const __m256i tt = _mm256_or_si256(tt3, _mm256_set_epi64x(pos0, pos1, pos2, pos3));

	constexpr uint64_t mask_1_transpose = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2_transpose = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3_transpose = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c_transpose = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d_transpose = 0b00001'00000'00000'00000'10000ULL;

	__m256i t, s;
	__m256i c = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_c_transpose))));
	__m256i d = _mm256_and_si256(tt, _mm256_set1_epi64x(int64_t(X2(mask_d_transpose))));

	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 4)), _mm256_set1_epi64x(int64_t(X2(mask_1_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 4));
	t = _mm256_and_si256(_mm256_xor_si256(c, _mm256_srli_epi64(c, 8)), _mm256_set1_epi64x(int64_t(X2(mask_2_transpose))));
	c = _mm256_xor_si256(_mm256_xor_si256(c, t), _mm256_slli_epi64(t, 8));
	s = _mm256_and_si256(_mm256_xor_si256(d, _mm256_srli_epi64(d, 16)), _mm256_set1_epi64x(int64_t(X2(mask_3_transpose))));
	d = _mm256_xor_si256(_mm256_xor_si256(d, s), _mm256_slli_epi64(s, 16));

	const int64_t pos0t = int64_t(transpose_5x5_table[pos0]);
	const int64_t pos1t = int64_t(transpose_5x5_table[pos1]);
	const int64_t pos2t = int64_t(transpose_5x5_table[pos2]);
	const int64_t pos3t = int64_t(transpose_5x5_table[pos3]);

	const __m256i zz = _mm256_or_si256(c, _mm256_or_si256(d, _mm256_set_epi64x(pos0t, pos1t, pos2t, pos3t)));

	const __m256i a1 = _mm256_sub_epi64(zz, tt);
	const __m256i a2 = _mm256_srai_epi32(a1, 32);
	const __m256i a3 = _mm256_shuffle_epi32(a2, 0b11110101);
	const __m256i a4 = _mm256_blendv_epi8(tt, zz, a3);

	const __m128i a5 = _mm256_extracti128_si256(a4, 0);
	const __m128i a6 = _mm256_extracti128_si256(a4, 1);

	const __m128i a7 = _mm_sub_epi64(a5, a6);
	const __m128i a8 = _mm_srai_epi32(a7, 32);
	const __m128i a9 = _mm_shuffle_epi32(a8, 0b11110101);
	const __m128i aa = _mm_blendv_epi8(a6, a5, a9);

	alignas(32) uint64_t result[2] = {};
	_mm_storeu_si128((__m128i*)result, aa);

	result[0] = std::min(result[0], result[1]);
	return result[0];

	//alignas(32) uint64_t result1[4] = {}, result2[4] = {};
	//_mm256_storeu_si256((__m256i*)result1, tt);
	//_mm256_storeu_si256((__m256i*)result2, zz);

	//for (int i = 0; i < 4; ++i) {
	//	result1[i] = std::min(result1[i], result2[i]);
	//}
	//for (int i = 1; i < 4; ++i) {
	//	result1[0] = std::min(result1[0], result1[i]);
	//}
	//return result1[0];

	//const bool b00 = result1[0] < result2[0];
	//const uint64_t r0 = b00 ? result1[0] : result2[0];
	//const bool b01 = result1[1] < result2[1];
	//const uint64_t r1 = b01 ? result1[1] : result2[1];
	//const bool b02 = result1[2] < result2[2];
	//const uint64_t r2 = b02 ? result1[2] : result2[2];
	//const bool b03 = result1[3] < result2[3];
	//const uint64_t r3 = b03 ? result1[3] : result2[3];
	//const bool b10 = r0 < r1;
	//const uint64_t r4 = b10 ? r0 : r1;
	//const bool b11 = r2 < r3;
	//const uint64_t r5 = b11 ? r2 : r3;
	//const bool b20 = r4 < r5;
	//return b20 ? r4 : r5;
}

void unittest() {
	std::mt19937 rnd(12345);

	for (int i = 0; i < 10000; ++i) {
		const uint64_t x = ((rnd() % (1ULL << 50)) << 5) + (rnd() % 25);

		const uint64_t y1 = code_unique1(x);
		const uint64_t y2 = code_unique2(x);
		const uint64_t y3 = code_unique3(x);
		const uint64_t y4 = code_unique4(x);
		const uint64_t y5 = code_unique5(x);
		assert(y1 == y2 && y1 == y3 && y1 == y4 && y1 == y5);
	}
}

inline uint64_t xorshift64(uint64_t x) {
	x = x ^ (x << 7);
	return x ^ (x >> 9);
}

#define DEF_BENCH_T(name) \
void bench_t_##name() {\
	std::cout << "Bench throughput:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		const uint64_t x = ((a % (1ULL << 50)) << 5) + (a % 25);\
		result ^= code_unique##name(x);\
		a = xorshift64(a);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

#define DEF_BENCH_L(name) \
void bench_l_##name() {\
	std::cout << "Bench latency:"#name << std::endl;\
	uint64_t result = 0;\
	uint64_t a = 0x1111222233334444ULL;\
	auto start = std::chrono::system_clock::now();\
	for (int i = 0; i < (1 << 30); ++i) {\
		const uint64_t x = ((a % (1ULL << 50)) << 5) + (a % 25);\
		result ^= code_unique##name(x);\
		a = xorshift64(result);\
	}\
	auto end = std::chrono::system_clock::now();\
	std::cout << "elapsed time = " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) << " ms" << std::endl;\
	std::cout << "result (for validation) = " << std::to_string(result) << std::endl << std::endl;\
}\

DEF_BENCH_T(1)
DEF_BENCH_T(2)
DEF_BENCH_T(3)
DEF_BENCH_T(4)
DEF_BENCH_T(5)

DEF_BENCH_L(1)
DEF_BENCH_L(2)
DEF_BENCH_L(3)
DEF_BENCH_L(4)
DEF_BENCH_L(5)

int main() {


	unittest();

	bench_l_1();
	bench_l_2();
	bench_l_3();
	bench_l_4();
	bench_l_5();

	bench_t_1();
	bench_t_2();
	bench_t_3();
	bench_t_4();
	bench_t_5();

	return 0;
}
