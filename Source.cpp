
#define _CRT_SECURE_NO_WARNINGS

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

#include <omp.h>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#endif

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <wmmintrin.h>
#include <immintrin.h>

//#include <boost/multiprecision/cpp_int.hpp>
//typedef boost::multiprecision::cpp_int Bigint;

//#define REP(i, a, b) for (int64_t i = (int64_t)(a); i < (int64_t)(b); i++)
//#define rep(i, a) REP(i, 0, a)
//#define ALL(a) (a.begin()), (a.end())


inline uint32_t bitscan_forward64(const uint64_t x, uint32_t *dest) {

	//xが非ゼロなら、立っているビットのうち最下位のものの位置をdestに代入して、非ゼロの値を返す。
	//xがゼロなら、ゼロを返す。このときのdestの値は未定義である。

#ifdef _MSC_VER
	return _BitScanForward64(reinterpret_cast<unsigned long *>(dest), x);
#else
	return x ? *dest = __builtin_ctzll(x), 1 : 0;
#endif

}

inline uint64_t pdep_intrinsics(uint64_t a, uint64_t mask) { return _pdep_u64(a, mask); }
inline uint64_t pext_intrinsics(uint64_t a, uint64_t mask) { return _pext_u64(a, mask); }

constexpr uint64_t BB_ALL_8X8_5X5 = 0b00011111'00011111'00011111'00011111'00011111ULL;
constexpr uint64_t BB_HORI_EDGE_8X8_5X5 = 0b00010001'00010001'00010001'00010001'00010001ULL;
constexpr uint64_t BB_VERT_EDGE_8X8_5X5 = 0b00011111'00000000'00000000'00000000'00011111ULL;


void visualize_ostle(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole) {

	assert((bb_player & bb_opponent) == 0);

	std::cout << std::endl;
	std::cout << "visualize ostle--------------------------" << std::endl;
	std::cout << std::endl;
	for (int i = 0; i < 5; ++i) {
		for (int j = 4; j >= 0; --j) {
			const uint64_t bb1 = 1ULL << (i * 8 + j);
			if (bb_player & bb1)std::cout << "■";
			else if (bb_opponent & bb1)std::cout << "□";
			else if (pos_hole == i * 5 + j)std::cout << "◎";
			else std::cout << "＋";
		}
		switch (i) {
		case 0: std::cout << "  player   : ■"; break;
		case 1: std::cout << "  opponent : □"; break;
		case 2: std::cout << "  hole     : ◎"; break;
		case 3: std::cout << "  empty    : ＋"; break;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "-----------------------------------------" << std::endl;

}


//OstleのBitboardは、8x8 bitboardの下位桁5x5部分を用いて表現することにした。
//理由は、以下の転置と反転の関数について、オセロのコードを流用できて楽だからである。

//しかし最速ではないかもしれない。例えばplayerとopponentの両方を同時に(転置|反転)変換したいとき、25bitに詰めてあれば2つ一気に変換可能で、そのほうが速いかもしれない。

uint64_t transpose_5x5_bitboard(uint64_t b) {
	//5x5 Bitboardが、8x8 bitboardの下位桁5x5部分を用いて表現されているとする。転置して返す。

	uint64_t t;

	t = (b ^ (b >> 7)) & 0x00AA00AA00AA00AAULL;
	b = b ^ t ^ (t << 7);
	t = (b ^ (b >> 14)) & 0x0000CCCC0000CCCCULL;
	b = b ^ t ^ (t << 14);
	t = (b ^ (b >> 28)) & 0x00000000F0F0F0F0ULL;
	b = b ^ t ^ (t << 28);

	return b;
}
uint64_t vertical_mirror_5x5_bitboard(uint64_t b) {
	//5x5 Bitboardが、8x8 bitboardの下位桁5x5部分を用いて表現されているとする。縦方向に反転させて返す。

	b = ((b >> 8) & 0x00FF00FF00FF00FFULL) | ((b << 8) & 0xFF00FF00FF00FF00ULL);
	b = ((b >> 16) & 0x0000FFFF0000FFFFULL) | ((b << 16) & 0xFFFF0000FFFF0000ULL);
	b = ((b >> 32) & 0x00000000FFFFFFFFULL) | ((b << 32) & 0xFFFFFFFF00000000ULL);

	return b >> 24;
}
uint64_t horizontal_mirror_5x5_bitboard(uint64_t b) {
	//5x5 Bitboardが、8x8 bitboardの下位桁5x5部分を用いて表現されているとする。横方向に反転させて返す。

	b = ((b >> 1) & 0x5555555555555555ULL) | ((b << 1) & 0xAAAAAAAAAAAAAAAAULL);
	b = ((b >> 2) & 0x3333333333333333ULL) | ((b << 2) & 0xCCCCCCCCCCCCCCCCULL);
	b = ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((b << 4) & 0xF0F0F0F0F0F0F0F0ULL);

	return b >> 3;
}

alignas(32) const static uint8_t transpose_5x5_table[32] = { 0,5,10,15,20,1,6,11,16,21,2,7,12,17,22,3,8,13,18,23,4,9,14,19,24,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

uint64_t transpose_ostle_5x5_bitboard(const uint64_t b) {
	//最下位5bitに穴の位置が格納され、次の50bitに5x5 Bitboard2つが25bitずつ詰めて格納されているとする。3要素すべてを転置して返す。

	constexpr uint64_t mask_1 = 0b00000'10100'00010'10000'01010ULL;
	constexpr uint64_t mask_2 = 0b00000'00000'11000'11100'01100ULL;
	constexpr uint64_t mask_3 = 0b00000'00000'00000'00000'10000ULL;

	constexpr uint64_t mask_c = 0b11110'11111'11111'11111'01111ULL;
	constexpr uint64_t mask_d = 0b00001'00000'00000'00000'10000ULL;

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	uint64_t t, s, c = b & X2(mask_c), d = b & X2(mask_d);

	t = (c ^ (c >> 4)) & X2(mask_1);
	c = c ^ t ^ (t << 4);
	t = (c ^ (c >> 8)) & X2(mask_2);
	c = c ^ t ^ (t << 8);
	s = (d ^ (d >> 16)) & X2(mask_3);
	d = d ^ s ^ (s << 16);

	return c | d | uint64_t(transpose_5x5_table[b % 32]);
}

alignas(32) const static uint8_t vertical_mirror_5x5_table[32] = { 20,21,22,23,24,15,16,17,18,19,10,11,12,13,14,5,6,7,8,9,0,1,2,3,4,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

uint64_t vertical_mirror_ostle_5x5_bitboard(const uint64_t b) {
	//最下位5bitに穴の位置が格納され、次の50bitに5x5 Bitboard2つが25bitずつ詰めて格納されているとする。3要素すべてを縦方向に反転させて返す。

	constexpr uint64_t mask_1_lo = 0b00000'00000'11111'00000'11111ULL;
	constexpr uint64_t mask_1_hi = 0b00000'11111'00000'11111'00000ULL;
	constexpr uint64_t mask_2_lo = 0b00000'00000'00000'11111'11111ULL;
	constexpr uint64_t mask_2_hi = 0b00000'11111'11111'00000'00000ULL;
	constexpr uint64_t mask_3_lo = 0b00000'00000'00000'00000'11111ULL;
	constexpr uint64_t mask_3_hi = 0b11111'11111'11111'11111'00000ULL;

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	uint64_t t;
	t = ((b >> 5) & X2(mask_1_lo)) | ((b << 5) & X2(mask_1_hi));
	t = ((t >> 10) & X2(mask_2_lo)) | ((t << 10) & X2(mask_2_hi));
	t = ((b >> 20) & X2(mask_3_lo)) | ((t << 5) & X2(mask_3_hi));

	return t | uint64_t(vertical_mirror_5x5_table[b % 32]);
}

alignas(32) const static uint8_t horizontal_mirror_5x5_table[32] = { 4,3,2,1,0,9,8,7,6,5,14,13,12,11,10,19,18,17,16,15,24,23,22,21,20,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };

uint64_t horizontal_mirror_ostle_5x5_bitboard(uint64_t b) {
	//最下位5bitに穴の位置が格納され、次の50bitに5x5 Bitboard2つが25bitずつ詰めて格納されているとする。3要素すべてを横方向に反転させて返す。

	constexpr uint64_t mask_1_lo = 0b00101'00101'00101'00101'00101ULL;
	constexpr uint64_t mask_1_hi = 0b01010'01010'01010'01010'01010ULL;
	constexpr uint64_t mask_2_lo = 0b00011'00011'00011'00011'00011ULL;
	constexpr uint64_t mask_2_hi = 0b01100'01100'01100'01100'01100ULL;
	constexpr uint64_t mask_3_lo = 0b00001'00001'00001'00001'00001ULL;
	constexpr uint64_t mask_3_hi = 0b11110'11110'11110'11110'11110ULL;

	constexpr auto X2 = [](const uint64_t x) {return (x | (x << 25)) << 5; };

	uint64_t t;
	t = ((b >> 1) & X2(mask_1_lo)) | ((b << 1) & X2(mask_1_hi));
	t = ((t >> 2) & X2(mask_2_lo)) | ((t << 2) & X2(mask_2_hi));
	t = ((b >> 4) & X2(mask_3_lo)) | ((t << 1) & X2(mask_3_hi));

	return t | uint64_t(horizontal_mirror_5x5_table[b % 32]);
}

uint64_t encode_ostle(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole) {
	//引数で受け取った盤面情報を、[0,2^55)の整数に可逆圧縮して返す。
	assert(pos_hole < 25);
	assert((bb_player & BB_ALL_8X8_5X5) == bb_player);
	assert((bb_opponent & BB_ALL_8X8_5X5) == bb_opponent);

	uint64_t answer = pos_hole;
	answer |= pext_intrinsics(bb_player, BB_ALL_8X8_5X5) << 30;
	answer |= pext_intrinsics(bb_opponent, BB_ALL_8X8_5X5) << 5;

	return answer;
}
void decode_ostle(const uint64_t code, uint64_t &bb_player, uint64_t &bb_opponent, uint64_t &pos_hole) {
	//encode_ostleで圧縮された値codeを受け取り、盤面を復元して残り3つの引数に代入する。
	assert(code < (1ULL << 55));
	assert((code % 32) < 25);

	pos_hole = code % 32;
	bb_player = pdep_intrinsics(code >> 30, BB_ALL_8X8_5X5);
	bb_opponent = pdep_intrinsics(code >> 5, BB_ALL_8X8_5X5);
}

uint64_t code_symmetry_naive(const uint32_t s, uint64_t code) {
	//encode_ostleで圧縮された値codeを、対称な盤面に変化させる。
	//変化させる方法は7通りあるが、引数sの下位3bitで指定される。

	uint64_t bb1 = 0, bb2 = 0, pos = 0;
	decode_ostle(code, bb1, bb2, pos);

	if (s & 1) {
		bb1 = horizontal_mirror_5x5_bitboard(bb1);
		bb2 = horizontal_mirror_5x5_bitboard(bb2);
		pos = horizontal_mirror_5x5_table[pos];
	}
	if (s & 2) {
		bb1 = vertical_mirror_5x5_bitboard(bb1);
		bb2 = vertical_mirror_5x5_bitboard(bb2);
		pos = vertical_mirror_5x5_table[pos];
	}
	if (s & 4) {
		bb1 = transpose_5x5_bitboard(bb1);
		bb2 = transpose_5x5_bitboard(bb2);
		pos = transpose_5x5_table[pos];
	}

	return encode_ostle(bb1, bb2, pos);
}

uint64_t code_symmetry(const uint32_t s, uint64_t code) {
	//encode_ostleで圧縮された値codeを、対称な盤面に変化させる。
	//変化させる方法は7通りあるが、引数sの下位3bitで指定される。

	if (s & 1)code = horizontal_mirror_ostle_5x5_bitboard(code);
	if (s & 2)code = vertical_mirror_ostle_5x5_bitboard(code);
	if (s & 4)code = transpose_ostle_5x5_bitboard(code);

	return code;
}

uint64_t code_unique_naive(const uint64_t code) {
	//encode_ostleで圧縮された値codeを、対称な盤面に変化させてもよいとしたとき、
	//変換後のcodeを整数として見たときの値が最も小さくなるような変換がどれか調べて、それを作用させた結果のcodeを返す。
	//これは対称な局面を同一視する操作そのものである。

	uint64_t answer = code;

	for (uint32_t i = 1; i <= 7; ++i) {
		const uint64_t new_code = code_symmetry_naive(i, code);
		answer = std::min(answer, new_code);
	}

	return answer;
}

uint64_t code_unique(const uint64_t code) {
	//encode_ostleで圧縮された値codeを、対称な盤面に変化させてもよいとしたとき、
	//変換後のcodeを整数として見たときの値が最も小さくなるような変換がどれか調べて、それを作用させた結果のcodeを返す。
	//これは対称な局面を同一視する操作そのものである。

	uint64_t answer = code;

	for (uint32_t i = 1; i <= 7; ++i) {
		const uint64_t new_code = code_symmetry(i, code);
		answer = std::min(answer, new_code);
	}

	return answer;
}

bool test_bitboard_symmetry(const uint64_t seed, const int length) {

	std::mt19937_64 rnd(seed);
	std::uniform_int_distribution<uint64_t>pos_dist(0, 24);

	const auto test_func = [&](const uint64_t bb1, const uint64_t bb2, const uint64_t pos, const auto func_5x5_bitboard, const auto func_ostle_5x5_bitboard) {

		const uint64_t code = encode_ostle(bb1, bb2, pos);

		uint64_t dec_bb1 = 0, dec_bb2 = 0, dec_pos = 0;
		decode_ostle(code, dec_bb1, dec_bb2, dec_pos);

		if (bb1 != dec_bb1 || bb2 != dec_bb2 || pos != dec_pos) {
			return false;
		}

		const uint64_t bb_pos = pdep_intrinsics(1ULL << pos, BB_ALL_8X8_5X5);
		const uint64_t bb_pos_converted = func_5x5_bitboard(bb_pos);
		uint32_t pos_converted = 0;
		bitscan_forward64(pext_intrinsics(bb_pos_converted, BB_ALL_8X8_5X5), &pos_converted);

		const uint64_t bb1_converted = func_5x5_bitboard(bb1);
		const uint64_t bb2_converted = func_5x5_bitboard(bb2);
		const uint64_t code_converted = func_ostle_5x5_bitboard(code);

		decode_ostle(code_converted, dec_bb1, dec_bb2, dec_pos);

		if (dec_bb1 != bb1_converted || dec_bb2 != bb2_converted || dec_pos != pos_converted) {
			return false;
		}

		if (bb1 != func_5x5_bitboard(bb1_converted) ||
			bb2 != func_5x5_bitboard(bb2_converted) ||
			code != func_ostle_5x5_bitboard(code_converted)) {
			return false;
		}

		const uint64_t unique1 = code_unique(code);
		const uint64_t unique2 = code_unique_naive(code);
		if (unique1 != unique2) {
			return false;
		}

		return true;
	};

	for (int i = 0; i < length; ++i) {

		if (i % (length / 10) == 0) {
			std::cout << "test_bitboard_symmetry: " << i << " / " << length << std::endl;
		}

		const uint64_t pos = pos_dist(rnd);
		uint64_t bb_occupied = pdep_intrinsics(1ULL << pos, BB_ALL_8X8_5X5);

		const auto fill_func = [&]() {
			uint64_t bb_answer = 0;
			for (int j = 4 + (rnd() % 2); j > 0;) {
				const uint64_t bb_rnd = pdep_intrinsics(1ULL << pos_dist(rnd), BB_ALL_8X8_5X5);
				if (bb_rnd & bb_occupied)continue;
				bb_answer |= bb_rnd;
				bb_occupied |= bb_rnd;
				--j;
			}
			return bb_answer;
		};

		const uint64_t bb1 = fill_func();
		const uint64_t bb2 = fill_func();


		if (!test_func(bb1, bb2, pos, transpose_5x5_bitboard, transpose_ostle_5x5_bitboard) ||
			!test_func(bb1, bb2, pos, vertical_mirror_5x5_bitboard, vertical_mirror_ostle_5x5_bitboard) ||
			!test_func(bb1, bb2, pos, horizontal_mirror_5x5_bitboard, horizontal_mirror_ostle_5x5_bitboard)) {
			std::cout << "test failed." << std::endl;
			visualize_ostle(bb1, bb2, pos);
			return false;
		}
	}
	std::cout << "test clear!" << std::endl;
	return true;
}

typedef std::array<uint8_t, 32> Moves; //下位5bitは着手位置、上位2bitは動かす方向。[0]に指し手の数、指し手自体は[1]から。最大24通り。

//4方向のことを「上下左右」とは呼ばず、「横方向と縦方向」「プラス方向とマイナス方向」のどちらかで呼ぶことにする。
//例えば「上方向」の代わりに「縦マイナス方向」と呼ぶことにする。
//横方向なら上位ビットが立っている。プラス方向なら下位ビットが立っている。

constexpr uint64_t pos_diff[4] = { uint64_t(int64_t(-5)), uint64_t(5), uint64_t(int64_t(-1)), uint64_t(1) };

constexpr uint8_t oneline_bb2index[33] =
{
	0,
	1,2,0,3,0,0,0,4,0,0,
	0,0,0,0,0,5,0,0,0,0,
	0,0,0,0,0,0,0,0,0,0,
	0,6
};

constexpr uint64_t BB_ONELINE_HORIZONTAL_8X8_5X5[5] = {
	0x0000'0000'0000'001FULL,
	0x0000'0000'0000'1F00ULL,
	0x0000'0000'001F'0000ULL,
	0x0000'0000'1F00'0000ULL,
	0x0000'001F'0000'0000ULL
};
constexpr uint64_t BB_ONELINE_VERTICAL_8X8_5X5[5] = {
	0x0000'0001'0101'0101ULL,
	0x0000'0002'0202'0202ULL,
	0x0000'0004'0404'0404ULL,
	0x0000'0008'0808'0808ULL,
	0x0000'0010'1010'1010ULL
};

//[a][b][c][d][e][f] a:プラス方向かマイナス方向か b:着手位置 c:穴の有無と位置 d:playerのbitboard e:opponentのbitboard f:着手後のplayer,opponentのbbと補助情報
uint8_t do_move_table[2][5][6][32][32][3] = {};//2*5*6*32*32*3 = 184,320 

//[a][b][c][d][e][f][g] a:プラス方向かマイナス方向か b:着手位置 c:穴の有無と位置 d:playerのbitboard e:opponentのbitboard f:補助情報 g:着手前のplayer,opponentのbb
uint8_t undo_move_table[2][5][6][32][32][6][2] = {};//2*5*6*32*32*6*2 = 737,280


void init_move_tables() {
	//do_move_tableとundo_move_tableを構築する。プログラム開始時に呼ぶ必要がある。

	//冪等性を保つためにここで全部ゼロ埋めする。
	for (int a = 0; a < 2; ++a) {
		for (int b = 0; b < 5; ++b) {
			for (int c = 0; c < 6; ++c) {
				for (int d = 0; d < 32; ++d) {
					for (int e = 0; e < 32; ++e) {
						for (int f = 0; f < 3; ++f) {
							do_move_table[a][b][c][d][e][f] = 0;
						}
						for (int f = 0; f < 6; ++f) {
							undo_move_table[a][b][c][d][e][f][0] = 0;
							undo_move_table[a][b][c][d][e][f][1] = 0;
						}
					}
				}
			}
		}
	}

	for (uint64_t d = 0; d < 32; ++d)for (uint64_t e = 0; e < 32; ++e) {
		if (d & e)continue;
		const uint64_t bb_occupied = d | e;

		for (int c = 0; c < 6; ++c) {
			if (c != 0 && ((1ULL << (c - 1)) & bb_occupied) != 0)continue;
			const uint64_t bb_hole = c == 0 ? 0 : (1ULL << (c - 1));

			uint64_t bb_player = d;
			for (uint32_t b = 0; bitscan_forward64(bb_player, &b); bb_player &= bb_player - 1) {
				for (int a = 0; a < 2; ++a) {

					int board[5];

					enum {
						SELF = 1,
						ENEMY = 2,
						HOLE = 3,
						EMPTY = 0
					};

					for (int i = 0; i < 5; ++i) {
						const uint64_t bb_i = 1ULL << i;
						if (bb_i & d)board[i] = SELF;
						else if (bb_i & e)board[i] = ENEMY;
						else if (bb_i & bb_hole)board[i] = HOLE;
						else board[i] = EMPTY;
					}

					int after_board[5];
					for (int i = 0; i < 5; ++i)after_board[i] = board[i];

					assert(after_board[b] == SELF);
					after_board[b] = EMPTY;

					int fall = EMPTY, num_push = 0;

					for (int cursor = b, push = SELF;;) {

						const int next_cursor = cursor + (a == 1 ? 1 : -1);
						if (next_cursor < 0 || 5 <= next_cursor || after_board[next_cursor] == HOLE) {
							fall = push;
							break;
						}
						if (after_board[next_cursor] == EMPTY) {
							after_board[next_cursor] = push;
							break;
						}

						std::swap(after_board[next_cursor], push);
						++num_push;
						cursor = next_cursor;
					}

					const int auxiliary = num_push + (fall == ENEMY ? 1 : 0);

					int bb_after_player = 0, bb_after_opponent = 0;

					for (int i = 0; i < 5; ++i) {
						if (after_board[i] == SELF) {
							bb_after_player |= 1 << i;
						}
						else if (after_board[i] == ENEMY) {
							bb_after_opponent |= 1 << i;
						}
					}

					assert(0 <= bb_after_player && bb_after_player < 32);
					assert(0 <= bb_after_opponent && bb_after_opponent < 32);
					assert(0 <= auxiliary && auxiliary <= 5);

					do_move_table[a][b][c][d][e][0] = bb_after_player;
					do_move_table[a][b][c][d][e][1] = bb_after_opponent;
					do_move_table[a][b][c][d][e][2] = auxiliary;

					assert(undo_move_table[a][b][c][bb_after_player][bb_after_opponent][auxiliary][0] == 0);
					assert(undo_move_table[a][b][c][bb_after_player][bb_after_opponent][auxiliary][1] == 0);

					undo_move_table[a][b][c][bb_after_player][bb_after_opponent][auxiliary][0] = uint8_t(d);
					undo_move_table[a][b][c][bb_after_player][bb_after_opponent][auxiliary][1] = uint8_t(e);
				}
			}
		}
	}
}

void generate_moves(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole, Moves& moves) {
	//合法手を全列挙する。
	//直前の局面に戻る手は反則だが、ここでは気にせず生成する。自殺する手も生成することに注意。

	assert((bb_player & BB_ALL_8X8_5X5) == bb_player);
	assert((bb_opponent & BB_ALL_8X8_5X5) == bb_opponent);
	assert((bb_player & bb_opponent) == 0);
	assert(pos_hole < 25);
	assert(4 <= _mm_popcnt_u64(bb_player) && _mm_popcnt_u64(bb_player) <= 5);
	assert(4 <= _mm_popcnt_u64(bb_opponent) && _mm_popcnt_u64(bb_opponent) <= 5);

	moves[0] = 0;

	//コマを動かす手を生成する。

	uint64_t b = pext_intrinsics(bb_player, BB_ALL_8X8_5X5);
	for (uint32_t x = 0; bitscan_forward64(b, &x); b &= b - 1) {
		for (uint32_t i = 0; i < 128; i += 32) {
			moves[++moves[0]] = uint8_t(x + i);
		}
	}

	//穴を動かす手を生成する。

	const uint64_t bb_empty = BB_ALL_8X8_5X5 ^ (bb_player | bb_opponent);
	const uint64_t bb_hole = pdep_intrinsics(1ULL << pos_hole, BB_ALL_8X8_5X5);

	//縦マイナス方向
	if ((bb_hole >> 8) & bb_empty) {
		moves[++moves[0]] = uint8_t(pos_hole);
	}
	//縦プラス方向
	if ((bb_hole << 8) & bb_empty) {
		moves[++moves[0]] = uint8_t(pos_hole + 32);
	}
	//横マイナス方向
	if ((bb_hole >> 1) & bb_empty) {
		moves[++moves[0]] = uint8_t(pos_hole + 64);
	}
	//横プラス方向
	if ((bb_hole << 1) & bb_empty) {
		moves[++moves[0]] = uint8_t(pos_hole + 96);
	}

	assert(moves[0] <= 24);
}

uint8_t do_move(uint64_t &bb_player, uint64_t &bb_opponent, uint64_t &pos_hole, const uint8_t move) {
	//moveを指して、他の引数を変更する。コマをいくつ押したかと場外に何が落ちたかを意味する補助情報を返り値とする。補助情報はundoのとき必要になる。

	assert(move % 32 < 25 && move / 32 < 4);

	//穴を動かす手の場合。合法手であることは生成関数が保証しているので、ただ動かして終了。
	if (pos_hole == (move % 32)) {
		pos_hole += pos_diff[move / 32];
		return 0xFF;
	}

	const uint64_t bb_hole = pdep_intrinsics(1ULL << pos_hole, BB_ALL_8X8_5X5);

	if (move / 64) {

		//横方向に動かす手の場合

		const int index1_dir = (move / 32) % 2;
		const int index2_pos = ((move % 32) % 5);
		const int y_pos = (move % 32) / 5;
		const int offset_bb = y_pos * 8;
		const uint8_t index3_hole = oneline_bb2index[(bb_hole >> offset_bb) % 256];
		const uint64_t index4_player = (bb_player >> offset_bb) % 256;
		const uint64_t index5_opponent = (bb_opponent >> offset_bb) % 256;

		assert(0 <= index1_dir && index1_dir < 2);
		assert(0 <= index2_pos && index2_pos < 5);
		assert(0 <= y_pos && y_pos < 5);
		assert(index3_hole < 6);
		assert(index4_player < 32);
		assert(index5_opponent < 32);

		bb_player =
			(bb_player & (BB_ALL_8X8_5X5 ^ BB_ONELINE_HORIZONTAL_8X8_5X5[y_pos])) |
			(uint64_t(do_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][0]) << offset_bb);

		bb_opponent =
			(bb_opponent & (BB_ALL_8X8_5X5 ^ BB_ONELINE_HORIZONTAL_8X8_5X5[y_pos])) |
			(uint64_t(do_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][1]) << offset_bb);

		return do_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][2];
	}
	else {

		//縦方向に動かす手の場合

		const int index1_dir = (move / 32) % 2;
		const int index2_pos = ((move % 32) / 5);
		const int x_pos = (move % 32) % 5;
		const uint8_t index3_hole = oneline_bb2index[pext_intrinsics(bb_hole, BB_ONELINE_VERTICAL_8X8_5X5[x_pos])];
		const uint64_t index4_player = pext_intrinsics(bb_player, BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);
		const uint64_t index5_opponent = pext_intrinsics(bb_opponent, BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);

		assert(0 <= index1_dir && index1_dir < 2);
		assert(0 <= index2_pos && index2_pos < 5);
		assert(0 <= x_pos && x_pos < 5);
		assert(index3_hole < 6);
		assert(index4_player < 32);
		assert(index5_opponent < 32);

		bb_player =
			(bb_player & (BB_ALL_8X8_5X5 ^ BB_ONELINE_VERTICAL_8X8_5X5[x_pos])) |
			pdep_intrinsics(do_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][0], BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);

		bb_opponent =
			(bb_opponent & (BB_ALL_8X8_5X5 ^ BB_ONELINE_VERTICAL_8X8_5X5[x_pos])) |
			pdep_intrinsics(do_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][1], BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);

		return do_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][2];
	}

	assert(false);
	return 0;
}

void undo_move(uint64_t &bb_player, uint64_t &bb_opponent, uint64_t &pos_hole, const uint8_t move, const uint8_t auxiliary) {
	//moveを指したあとの状態だと仮定してmoveを指す前に戻す。

	assert(move % 32 < 25 && move / 32 < 4);
	assert(auxiliary < 6 || auxiliary == 0xFF);

	//穴を動かす手の場合。合法手であることは生成関数が保証しているので、ただ動かして終了。
	if (auxiliary == 0xFF) {
		pos_hole -= pos_diff[move / 32];
		return;
	}

	const uint64_t bb_hole = pdep_intrinsics(1ULL << pos_hole, BB_ALL_8X8_5X5);

	if (move / 64) {

		//横方向に動かす手の場合

		const int index1_dir = (move / 32) % 2;
		const int index2_pos = ((move % 32) % 5);
		const int y_pos = (move % 32) / 5;
		const int offset_bb = y_pos * 8;
		const uint8_t index3_hole = oneline_bb2index[(bb_hole >> offset_bb) % 256];
		const uint64_t index4_player = (bb_player >> offset_bb) % 256;
		const uint64_t index5_opponent = (bb_opponent >> offset_bb) % 256;

		assert(0 <= index1_dir && index1_dir < 2);
		assert(0 <= index2_pos && index2_pos < 5);
		assert(0 <= y_pos && y_pos < 5);
		assert(index3_hole < 6);
		assert(index4_player < 32);
		assert(index5_opponent < 32);

		bb_player =
			(bb_player & (BB_ALL_8X8_5X5 ^ BB_ONELINE_HORIZONTAL_8X8_5X5[y_pos])) |
			(uint64_t(undo_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][auxiliary][0]) << offset_bb);

		bb_opponent =
			(bb_opponent & (BB_ALL_8X8_5X5 ^ BB_ONELINE_HORIZONTAL_8X8_5X5[y_pos])) |
			(uint64_t(undo_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][auxiliary][1]) << offset_bb);

		return;
	}
	else {

		//縦方向に動かす手の場合

		const int index1_dir = (move / 32) % 2;
		const int index2_pos = ((move % 32) / 5);
		const int x_pos = (move % 32) % 5;
		const uint8_t index3_hole = oneline_bb2index[pext_intrinsics(bb_hole, BB_ONELINE_VERTICAL_8X8_5X5[x_pos])];
		const uint64_t index4_player = pext_intrinsics(bb_player, BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);
		const uint64_t index5_opponent = pext_intrinsics(bb_opponent, BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);

		assert(0 <= index1_dir && index1_dir < 2);
		assert(0 <= index2_pos && index2_pos < 5);
		assert(0 <= x_pos && x_pos < 5);
		assert(index3_hole < 6);
		assert(index4_player < 32);
		assert(index5_opponent < 32);

		bb_player =
			(bb_player & (BB_ALL_8X8_5X5 ^ BB_ONELINE_VERTICAL_8X8_5X5[x_pos])) |
			pdep_intrinsics(undo_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][auxiliary][0], BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);

		bb_opponent =
			(bb_opponent & (BB_ALL_8X8_5X5 ^ BB_ONELINE_VERTICAL_8X8_5X5[x_pos])) |
			pdep_intrinsics(undo_move_table[index1_dir][index2_pos][index3_hole][index4_player][index5_opponent][auxiliary][1], BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);

		return;
	}
}

bool test_move(const uint64_t seed, const int length) {
	//do_moveの直後にundo_moveしたらもとに戻ることを確認する。

	std::mt19937_64 rnd(seed);
	std::uniform_int_distribution<uint64_t>pos_dist(0, 24);

	for (int i = 0; i < length; ++i) {
		if (i % (length / 10) == 0) {
			std::cout << "test_move: " << i << " / " << length << std::endl;
		}

		const uint64_t pos = pos_dist(rnd);
		uint64_t bb_occupied = pdep_intrinsics(1ULL << pos, BB_ALL_8X8_5X5);

		const auto fill_func = [&]() {
			uint64_t bb_answer = 0;
			for (int j = 4 + (rnd() % 2); j > 0;) {
				const uint64_t bb_rnd = pdep_intrinsics(1ULL << pos_dist(rnd), BB_ALL_8X8_5X5);
				if (bb_rnd & bb_occupied)continue;
				bb_answer |= bb_rnd;
				bb_occupied |= bb_rnd;
				--j;
			}
			return bb_answer;
		};

		const uint64_t bb1 = fill_func();
		const uint64_t bb2 = fill_func();

		Moves moves;
		generate_moves(bb1, bb2, pos, moves);
		for (int j = 1; j <= moves[0]; ++j) {
			uint64_t bb1a = bb1, bb2a = bb2, pos_a = pos;
			const auto auxiliary = do_move(bb1a, bb2a, pos_a, moves[j]);
			uint64_t bb1b = bb1a, bb2b = bb2a, pos_b = pos_a;
			undo_move(bb1b, bb2b, pos_b, moves[j], auxiliary);

			if (bb1 != bb1b || bb2 != bb2b || pos != pos_b) {
				std::cout << "test failed." << std::endl;
				visualize_ostle(bb1, bb2, pos);
				return false;
			}
		}
	}
	std::cout << "test clear!" << std::endl;
	return true;
}



bool is_checkmate_naive(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole, const bool visualize = false) {
	//player(手番側)にある指し手cが存在して、cを指すと相手のコマが3個になる⇔即勝利局面である⇔trueを返す。

	if (_mm_popcnt_u64(bb_opponent) != 4)return false;

	Moves moves;
	generate_moves(bb_player, bb_opponent, pos_hole, moves);

	for (int i = 1; i <= moves[0]; ++i) {
		uint64_t next_bb_player = bb_player, next_bb_opponent = bb_opponent, next_pos_hole = pos_hole;
		do_move(next_bb_player, next_bb_opponent, next_pos_hole, moves[i]);
		if (visualize) {
			std::cout << "is_checkmate_naive: i = " << i << ", move = " << int(moves[i]) << std::endl;
			visualize_ostle(next_bb_player, next_bb_opponent, next_pos_hole);
		}
		if (_mm_popcnt_u64(next_bb_opponent) == 3) {
			return true;
		}
	}

	return false;
}

bool is_checkmate_slow(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole) {
	//player(手番側)にある指し手cが存在して、cを指すと相手のコマが3個になる⇔即勝利局面である⇔trueを返す。

	if (_mm_popcnt_u64(bb_opponent) != 4)return false;

	const uint64_t bb_hole = pdep_intrinsics(1ULL << pos_hole, BB_ALL_8X8_5X5);

	//key observation:
	//相手のコマを横プラス方向に押し出すことができる⇔自分のコマのbitboardを1bit左シフトして、相手のコマのbitboardに加算すれば、繰り上がりにより場外のビットが立つ。
	//(ただし左シフトした瞬間に場外に出る自分のコマは加算の前に除外しておく必要がある)
	{
		const uint64_t bb_outside = (~BB_ALL_8X8_5X5) | bb_hole;
		if ((((bb_player << 1) & (~bb_outside)) + bb_opponent) & bb_outside) {
			return true;
		}
	}

	//縦方向および横マイナス方向への押し出しに関しては、bitboardを水平反転・転置してから同様に計算できる。

	//縦プラス方向について。（転置）
	const uint64_t bb_p1 = transpose_5x5_bitboard(bb_player);
	const uint64_t bb_o1 = transpose_5x5_bitboard(bb_opponent);
	const uint64_t bb_h1 = transpose_5x5_bitboard(bb_hole);
	const uint64_t bb_outside1 = (~BB_ALL_8X8_5X5) | bb_h1;
	if ((((bb_p1 << 1) & (~bb_outside1)) + bb_o1) & bb_outside1) {
		return true;
	}

	//横マイナス方向について。（水平反転）
	const uint64_t bb_p2 = horizontal_mirror_5x5_bitboard(bb_player);
	const uint64_t bb_o2 = horizontal_mirror_5x5_bitboard(bb_opponent);
	const uint64_t bb_h2 = horizontal_mirror_5x5_bitboard(bb_hole);
	const uint64_t bb_outside2 = (~BB_ALL_8X8_5X5) | bb_h2;
	if ((((bb_p2 << 1) & (~bb_outside2)) + bb_o2) & bb_outside2) {
		return true;
	}

	//縦マイナス方向について。（転置してから水平反転）
	const uint64_t bb_p3 = horizontal_mirror_5x5_bitboard(bb_p1);
	const uint64_t bb_o3 = horizontal_mirror_5x5_bitboard(bb_o1);
	const uint64_t bb_h3 = horizontal_mirror_5x5_bitboard(bb_h1);
	const uint64_t bb_outside3 = (~BB_ALL_8X8_5X5) | bb_h3;
	if ((((bb_p3 << 1) & (~bb_outside3)) + bb_o3) & bb_outside3) {
		return true;
	}

	return false;
}

bool is_checkmate(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole) {
	//player(手番側)にある指し手cが存在して、cを指すと相手のコマが3個になる⇔即勝利局面である⇔trueを返す。

	if (_mm_popcnt_u64(bb_opponent) != 4)return false;

	const uint64_t bb_hole = pdep_intrinsics(1ULL << pos_hole, BB_ALL_8X8_5X5);
	//const uint64_t bb_floor = BB_ALL_8X8_5X5 ^ bb_hole;

	//(以下、盤面の端をedgeと呼ぶが、穴の隣とedgeを合わせてcliffと呼ぶ。)
	const uint64_t bb_hori_cliff = BB_HORI_EDGE_8X8_5X5 | (((bb_hole >> 1) | (bb_hole << 1)) & BB_ALL_8X8_5X5);
	const uint64_t bb_vert_cliff = BB_VERT_EDGE_8X8_5X5 | (((bb_hole >> 8) | (bb_hole << 8)) & BB_ALL_8X8_5X5);

	//相手のコマのうち、(横|縦)方向に押し出されうるコマ(＝真(横|縦)が場外であるような位置に居るコマ)のbitboardをつくる。

	const uint64_t bb_opponent_hori_cliff_1 = bb_hori_cliff & bb_opponent;
	const uint64_t bb_opponent_vert_cliff_1 = bb_vert_cliff & bb_opponent;

	//相手のコマのうち、(横|縦)方向に押し出されうるコマと(横|縦)方向に隣接しているコマを「(横|縦)方向の押し出しに加担しうる」と呼ぶことにする。
	//相手のコマのうち、「加担しうる」コマと隣接しているコマも「加担しうる」といえる。4連続まで再帰するとして、「加担しうる」コマのbitboardをつくる。
	const uint64_t bb_opponent_hori_cliff_2 = (bb_opponent_hori_cliff_1 | (bb_opponent_hori_cliff_1 << 1) | (bb_opponent_hori_cliff_1 >> 1)) & bb_opponent;
	const uint64_t bb_opponent_vert_cliff_2 = (bb_opponent_vert_cliff_1 | (bb_opponent_vert_cliff_1 << 8) | (bb_opponent_vert_cliff_1 >> 8)) & bb_opponent;
	const uint64_t bb_opponent_hori_cliff_3 = (bb_opponent_hori_cliff_2 | (bb_opponent_hori_cliff_2 << 1) | (bb_opponent_hori_cliff_2 >> 1)) & bb_opponent;
	const uint64_t bb_opponent_vert_cliff_3 = (bb_opponent_vert_cliff_2 | (bb_opponent_vert_cliff_2 << 8) | (bb_opponent_vert_cliff_2 >> 8)) & bb_opponent;
	const uint64_t bb_opponent_hori_cliff_4 = (bb_opponent_hori_cliff_3 | (bb_opponent_hori_cliff_3 << 1) | (bb_opponent_hori_cliff_3 >> 1)) & bb_opponent;
	const uint64_t bb_opponent_vert_cliff_4 = (bb_opponent_vert_cliff_3 | (bb_opponent_vert_cliff_3 << 8) | (bb_opponent_vert_cliff_3 >> 8)) & bb_opponent;

	//key observation:
	//相手のコマのうち「加担しうる」コマのいずれかに対して、自分のコマが隣接している⇔相手のコマを押し出す指し手が存在する。
	if (((bb_opponent_hori_cliff_4 << 1) | (bb_opponent_hori_cliff_4 >> 1)) & bb_player) {
		return true;
	}
	if (((bb_opponent_vert_cliff_4 << 8) | (bb_opponent_vert_cliff_4 >> 8)) & bb_player) {
		return true;
	}

	return false;
}

bool test_checkmate_detector_func(const uint64_t seed, const int length) {

	std::mt19937_64 rnd(seed);
	std::uniform_int_distribution<uint64_t>pos_dist(0, 24);

	for (int i = 0; i < length; ++i) {
		if (i % (length / 10) == 0) {
			std::cout << "test_checkmate_detector_func: " << i << " / " << length << std::endl;
		}

		const uint64_t pos = pos_dist(rnd);
		uint64_t bb_occupied = pdep_intrinsics(1ULL << pos, BB_ALL_8X8_5X5);

		const auto fill_func = [&]() {
			uint64_t bb_answer = 0;
			for (int j = 4 + (rnd() % 2); j > 0;) {
				const uint64_t bb_rnd = pdep_intrinsics(1ULL << pos_dist(rnd), BB_ALL_8X8_5X5);
				if (bb_rnd & bb_occupied)continue;
				bb_answer |= bb_rnd;
				bb_occupied |= bb_rnd;
				--j;
			}
			return bb_answer;
		};

		const uint64_t bb1 = fill_func();
		const uint64_t bb2 = fill_func();

		const bool b1 = is_checkmate_naive(bb1, bb2, pos);
		const bool b2 = is_checkmate_slow(bb1, bb2, pos);
		const bool b3 = is_checkmate(bb1, bb2, pos);

		if (b1 != b2 || b1 != b3) {
			std::cout << "test failed." << std::endl;
			visualize_ostle(bb1, bb2, pos);
			const bool b1_ = is_checkmate_naive(bb1, bb2, pos, true);
			const bool b2_ = is_checkmate_slow(bb1, bb2, pos);
			const bool b3_ = is_checkmate(bb1, bb2, pos);
			return false;
		}

	}
	std::cout << "test clear!" << std::endl;
	return true;
}

void encode_25numbers(const uint64_t code, std::string &dest) {
	//encode_ostleで圧縮された値codeを受け取り、盤面を25文字の可読なフォーマットに変換してdestに代入する。
	//5*5マスの盤面を一列に並べたとして、'1'は手番側のコマ、'2'は相手のコマ、'3'は穴、'0'は空白マスを意味する。

	const uint64_t pos_hole = code % 32;

	dest.clear();

	for (int i = 0; i < 25; ++i) {
		if (code & (1ULL << (i + 30))) {
			dest += "1";
		}
		else if (code & (1ULL << (i + 5))) {
			dest += "2";
		}
		else if (pos_hole == i) {
			dest += "3";
		}
		else {
			dest += "0";
		}
	}
}

void decode_25numbers(const std::string &s, uint64_t &dest) {
	//encode_25numbersで変換された文字列sを受け取り、盤面を25文字の可読なフォーマットに変換して返す。
	//5*5マスの盤面を一列に並べたとして、'1'は手番側のコマ、'2'は相手のコマ、'3'は穴、'0'は空白マスを意味する。

	dest = 0;
	for (uint64_t i = 0; i < 25; ++i) {
		if (s[i] == '0') {
			continue;
		}
		else if (s[i] == '1') {
			dest += 1ULL << (i + 30);
		}
		else if (s[i] == '2') {
			dest += 1ULL << (i + 5);
		}
		else if (s[i] == '3') {
			dest += i;
		}
	}
}

const char BASE64[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void encode_base64(const uint64_t code, std::string &dest) {
	dest.clear();
	if (code == 0) {
		dest.push_back(BASE64[0]);
		return;
	}
	for (uint64_t c = code; c; c /= 64) {
		dest.push_back(BASE64[c % 64]);
	}
}

void decode_base64(const std::string &s, uint64_t &dest) {
	dest = 0;
	for (uint64_t i = 0; i < s.size(); ++i) {
		if ('A' <= s[i] && s[i] <= 'Z') {
			dest += uint64_t(s[i] - 'A') << (i * 6);
		}
		else if ('a' <= s[i] && s[i] <= 'z') {
			dest += (uint64_t(s[i] - 'a') + 26ULL) << (i * 6);
		}
		else if ('0' <= s[i] && s[i] <= '9') {
			dest += (uint64_t(s[i] - '0') + 52ULL) << (i * 6);
		}
		else if (s[i] == '+') {
			dest += 62ULL << (i * 6);
		}
		else if (s[i] == '/') {
			dest += 63ULL << (i * 6);
		}
		else {
			throw "error: failed to decode base64 string.";
		}
	}
}

bool test_base64_func(const uint64_t seed, const int length) {

	std::mt19937_64 rnd(seed);
	std::uniform_int_distribution<uint64_t>pos_dist(0, 24);

	std::string code;

	for (int i = 0; i < length; ++i) {
		if (i % (length / 10) == 0) {
			std::cout << "test_base64_func: " << i << " / " << length << std::endl;
		}

		const uint64_t x = rnd();
		encode_base64(x, code);
		uint64_t y = 0;
		decode_base64(code, y);
		if (x != y) {
			std::cout << "test failed." << std::endl;
			return false;
		}
	}
	std::cout << "test clear!" << std::endl;
	return true;
}

std::string my_itos(const uint64_t i, const int width, const char fill) {
	//整数iをstringに変換する。桁数がwidth未満なら、文字fillを先頭にくっつけて桁数をwidthに揃える。

	const std::string number = std::to_string(i);
	std::string prefix;
	while (int(prefix.size() + number.size()) < width) {
		prefix.push_back(fill);
	}
	return prefix + number;
}


std::string get_datetime_str() {
	time_t t = time(nullptr);
	const tm* localTime = localtime(&t);

	std::string s;
	s += my_itos(uint64_t(1900 + localTime->tm_year), 4, '0') + "/";
	s += my_itos(uint64_t(localTime->tm_mon + 1), 2, '0') + "/";
	s += my_itos(uint64_t(localTime->tm_mday), 2, '0') + " ";
	s += my_itos(uint64_t(localTime->tm_hour), 2, '0') + ":";
	s += my_itos(uint64_t(localTime->tm_min), 2, '0') + ":";
	s += my_itos(uint64_t(localTime->tm_sec), 2, '0');

	return s;
}

class Encoder_AES {
	//ハッシュ関数としてAES暗号を流用する。

	//cf: https://gist.github.com/acapola/d5b940da024080dfaf5f

private:

	__m128i key_schedule[20];//the expanded key

	__m128i aes_128_key_expansion(__m128i key, __m128i keygened) const {
		keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3, 3, 3, 3));
		key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
		key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
		key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
		return _mm_xor_si128(key, keygened);
	}

	void aes128_load_keys() {
		key_schedule[1] = aes_128_key_expansion(key_schedule[0], _mm_aeskeygenassist_si128(key_schedule[0], 0x01));
		key_schedule[2] = aes_128_key_expansion(key_schedule[1], _mm_aeskeygenassist_si128(key_schedule[1], 0x02));
		key_schedule[3] = aes_128_key_expansion(key_schedule[2], _mm_aeskeygenassist_si128(key_schedule[2], 0x04));
		key_schedule[4] = aes_128_key_expansion(key_schedule[3], _mm_aeskeygenassist_si128(key_schedule[3], 0x08));
		key_schedule[5] = aes_128_key_expansion(key_schedule[4], _mm_aeskeygenassist_si128(key_schedule[4], 0x10));
		key_schedule[6] = aes_128_key_expansion(key_schedule[5], _mm_aeskeygenassist_si128(key_schedule[5], 0x20));
		key_schedule[7] = aes_128_key_expansion(key_schedule[6], _mm_aeskeygenassist_si128(key_schedule[6], 0x40));
		key_schedule[8] = aes_128_key_expansion(key_schedule[7], _mm_aeskeygenassist_si128(key_schedule[7], 0x80));
		key_schedule[9] = aes_128_key_expansion(key_schedule[8], _mm_aeskeygenassist_si128(key_schedule[8], 0x1B));
		key_schedule[10] = aes_128_key_expansion(key_schedule[9], _mm_aeskeygenassist_si128(key_schedule[9], 0x36));

		// generate decryption keys in reverse order.
		// k[10] is shared by last encryption and first decryption rounds
		// k[0] is shared by first encryption round and last decryption round (and is the original user key)
		// For some implementation reasons, decryption key schedule is NOT the encryption key schedule in reverse order

		key_schedule[11] = _mm_aesimc_si128(key_schedule[9]);
		key_schedule[12] = _mm_aesimc_si128(key_schedule[8]);
		key_schedule[13] = _mm_aesimc_si128(key_schedule[7]);
		key_schedule[14] = _mm_aesimc_si128(key_schedule[6]);
		key_schedule[15] = _mm_aesimc_si128(key_schedule[5]);
		key_schedule[16] = _mm_aesimc_si128(key_schedule[4]);
		key_schedule[17] = _mm_aesimc_si128(key_schedule[3]);
		key_schedule[18] = _mm_aesimc_si128(key_schedule[2]);
		key_schedule[19] = _mm_aesimc_si128(key_schedule[1]);
	}

	__m128i DO_ENC_BLOCK(__m128i m) const {
		m = _mm_xor_si128(m, key_schedule[0]);
		m = _mm_aesenc_si128(m, key_schedule[1]);
		m = _mm_aesenc_si128(m, key_schedule[2]);
		m = _mm_aesenc_si128(m, key_schedule[3]);
		m = _mm_aesenc_si128(m, key_schedule[4]);
		m = _mm_aesenc_si128(m, key_schedule[5]);
		m = _mm_aesenc_si128(m, key_schedule[6]);
		m = _mm_aesenc_si128(m, key_schedule[7]);
		m = _mm_aesenc_si128(m, key_schedule[8]);
		m = _mm_aesenc_si128(m, key_schedule[9]);
		m = _mm_aesenclast_si128(m, key_schedule[10]);
		return m;
	}

	//__m128i DO_DEC_BLOCK(__m128i m) const {
	//	m = _mm_xor_si128(m, key_schedule[10]);
	//	m = _mm_aesdec_si128(m, key_schedule[11]);
	//	m = _mm_aesdec_si128(m, key_schedule[12]);
	//	m = _mm_aesdec_si128(m, key_schedule[13]);
	//	m = _mm_aesdec_si128(m, key_schedule[14]);
	//	m = _mm_aesdec_si128(m, key_schedule[15]);
	//	m = _mm_aesdec_si128(m, key_schedule[16]);
	//	m = _mm_aesdec_si128(m, key_schedule[17]);
	//	m = _mm_aesdec_si128(m, key_schedule[18]);
	//	m = _mm_aesdec_si128(m, key_schedule[19]);
	//	m = _mm_aesdeclast_si128(m, key_schedule[0]);
	//	return m;
	//}

	//void aes128_load_key(const int8_t * const enc_key) {
	//	key_schedule[0] = _mm_loadu_si128((const __m128i*) enc_key);
	//	aes128_load_keys();
	//}

	void aes128_load_key(const uint64_t enc_key) {
		key_schedule[0] = _mm_cvtsi64_si128(enc_key);
		aes128_load_keys();
	}

public:

	//Encoder_AES(const int8_t * const enc_key) {
	//	aes128_load_key(enc_key);
	//}

	Encoder_AES(const uint64_t enc_key) {
		aes128_load_key(enc_key);
	}

	Encoder_AES() : Encoder_AES(0ULL) {}

	//void aes128_enc(int8_t *plain_text, int8_t *cipher_text) {
	//	const __m128i m = _mm_loadu_si128((__m128i *) plain_text);
	//	const __m128i n = DO_ENC_BLOCK(m);
	//	_mm_storeu_si128((__m128i *) cipher_text, n);
	//}

	//void aes128_dec(int8_t *cipher_text, int8_t *plain_text) {
	//	const __m128i m = _mm_loadu_si128((__m128i *) cipher_text);
	//	const __m128i n = DO_DEC_BLOCK(m);
	//	_mm_storeu_si128((__m128i *) plain_text, n);
	//}

	uint64_t aes128_enc(const uint64_t plain_text) const {
		const __m128i m = _mm_cvtsi64_si128(plain_text);
		const __m128i n = DO_ENC_BLOCK(m);
		return _mm_cvtsi128_si64(n);
	}

	__m128i aes128_enc(const uint64_t plain_text_1, const uint64_t plain_text_2) const {
		const __m128i m = _mm_set_epi64x(plain_text_1, plain_text_2);
		const __m128i n = DO_ENC_BLOCK(m);
		return n;
	}
};

class PositionEnumerator {

private:

	std::vector<uint64_t>positions;

	int num_start_piece;

	void dfs_position(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole, const int cursor, const int num_piece_player, const int num_piece_opponent) {

		const int num_remaining_object = num_piece_player + num_piece_opponent + (cursor <= pos_hole ? 1 : 0);

		if (cursor == 25) {
			assert(num_remaining_object == 0);
			const uint64_t code = encode_ostle(bb_player, bb_opponent, pos_hole);
			positions.push_back(code);
			return;
		}
		else {
			assert(num_remaining_object + cursor < 25 || pos_hole == cursor || num_piece_player + num_piece_opponent > 0);
		}

		if (pos_hole == cursor) {
			dfs_position(bb_player, bb_opponent, pos_hole, cursor + 1, num_piece_player, num_piece_opponent);
			return;
		}

		if (num_remaining_object + cursor < 25) {
			dfs_position(bb_player, bb_opponent, pos_hole, cursor + 1, num_piece_player, num_piece_opponent);
		}

		const uint64_t bb_cursor = pdep_intrinsics(1ULL << cursor, BB_ALL_8X8_5X5);

		if (num_piece_player > 0) {
			dfs_position(bb_player | bb_cursor, bb_opponent, pos_hole, cursor + 1, num_piece_player - 1, num_piece_opponent);
		}
		if (num_piece_opponent > 0) {
			dfs_position(bb_player, bb_opponent | bb_cursor, pos_hole, cursor + 1, num_piece_player, num_piece_opponent - 1);
		}
	}

	uint64_t dfs_position_root(const uint64_t pos_hole, const int num_piece_player, const int num_piece_opponent, std::vector<uint64_t> &all_positions) {
		//穴の位置とコマの数を引数に取り、その条件に沿った盤面を全列挙して、対称な局面を同一視して重複削除してからall_positionsの末尾に加える。返り値は、最終的に加えた盤面の数とする。

		positions.clear();
		positions.reserve(500'000'000ULL);

		dfs_position(0, 0, pos_hole, 0, num_piece_player, num_piece_opponent);

		const int64_t siz = positions.size();

#pragma omp parallel for
		for (int64_t i = 0; i < siz; ++i) {
			positions[i] = code_unique(positions[i]);
		}
		std::sort(positions.begin(), positions.end());

		const auto result = std::unique(positions.begin(), positions.end());
		positions.erase(result, positions.end());
		std::copy(positions.begin(), positions.end(), std::back_inserter(all_positions));

		std::cout << "result: dfs_position_root(" << my_itos(pos_hole, 2, ' ') << "," << num_piece_player << "," << num_piece_opponent << "): "
			<< my_itos(uint64_t(siz), 9, ' ') << " positions; " << my_itos(positions.size(), 9, ' ') << " unique positions." << std::endl;

		return positions.size();
	}

public:

	PositionEnumerator() {
		positions.reserve(500'000'000ULL);
	}

	void do_enumerate(const int start_piece, std::vector<uint64_t> &all_positions) {

		num_start_piece = start_piece;

		const uint64_t holes[6] = { 0,1,2,6,7,12 };
		uint64_t num_unique_positions = 0;

		std::cout << "LOG: [" << get_datetime_str() << "] start: dfs_position_root" << std::endl;

		switch (num_start_piece) {
		case 10:
			for (int i = 0; i < 6; ++i)num_unique_positions += dfs_position_root(holes[i], 5, 5, all_positions);
		case 9:
			for (int i = 0; i < 6; ++i)num_unique_positions += dfs_position_root(holes[i], 5, 4, all_positions);
			for (int i = 0; i < 6; ++i)num_unique_positions += dfs_position_root(holes[i], 4, 5, all_positions);
		case 8:
			for (int i = 0; i < 6; ++i)num_unique_positions += dfs_position_root(holes[i], 4, 4, all_positions);
			break;
		default:
			assert(false);
		}

		std::cout << "LOG: [" << get_datetime_str() << "] finish: dfs_position_root" << std::endl;

		std::cout << "result: total number of the unique positions = " << num_unique_positions << std::endl;

		assert(num_unique_positions == all_positions.size());

		std::cout << "LOG: [" << get_datetime_str() << "] start: sort and verify uniqueness" << std::endl;

		std::sort(all_positions.begin(), all_positions.end());

		for (uint64_t i = 1; i < all_positions.size(); ++i) {
			assert(all_positions[i - 1] < all_positions[i]);
		}

		std::cout << "LOG: [" << get_datetime_str() << "] finish: sort and verify uniqueness" << std::endl;

		return;
	}
};

class BitVector {

protected:

	std::vector<uint64_t>bits;

	uint64_t siz;

	uint64_t population;

public:

	BitVector(const uint64_t size) {

		//1要素増やす理由は、派生クラスのget_bb_position関数で便利だから。
		bits.resize(((size + 63ULL) / 64ULL) + 1);

		siz = size;

		population = 0;
	}

	BitVector() :BitVector(1) {}

	void set1(const uint64_t pos) {
		assert(pos < bits.size() * 64);
		if ((bits[pos / 64] & (1ULL << (pos % 64))) == 0)++population;
		bits[pos / 64] |= 1ULL << (pos % 64);
	}

	void set0_all() {
		for (uint64_t i = 0; i < bits.size(); ++i) {
			bits[i] = 0;
		}
		population = 0;
	}

	bool get(const uint64_t pos) {
		assert(pos < bits.size() * 64);
		return (bits[pos / 64] & (1ULL << (pos % 64))) != 0;
	}

	uint64_t get_size() {
		return siz;
	}

	uint64_t get_bb_position(const uint64_t index) {
		//ビットベクトルが25bitごとに区切られたものだとして、index番目の区切りに相当する25bitを返す。

		const uint64_t bindex = index * 25;
		const uint64_t shift_len = (bindex % 64);

		if (shift_len + 25 < 64) {
			return (bits[bindex / 64] >> shift_len) & ((1ULL << 25) - 1ULL);
		}

		return ((bits[bindex / 64] >> shift_len) + (bits[(bindex / 64) + 1] << (64 - shift_len))) & ((1ULL << 25) - 1ULL);

		//注意:符号なし64bit整数を64bit右シフトするとゼロになるかと期待してはいけない。未定義動作であるし、実際ならない。
	}

	uint64_t popcount() {
		//立っているビットの総数を返す。

		return population;
	}

	uint64_t rank1_naive(const uint64_t index) {
		//bitsの範囲[0,index)のなかで立っているビットの数を返す。

		assert(index < bits.size() * 64);

		uint64_t answer = 0;
		for (uint64_t i = 63; i < index; i += 64) {
			answer += _mm_popcnt_u64(bits[i / 64]);
		}
		answer += _mm_popcnt_u64(bits[index / 64] & ((1ULL << (index % 64)) - 1));
		return answer;
	}
};

class BitVector_nodes : public BitVector {

private:

	std::vector<uint64_t>large_block;
	std::vector<uint16_t>small_block;

public:

	BitVector_nodes(const uint64_t size) : BitVector(size) {
		large_block.clear();
		small_block.clear();
	}

	BitVector_nodes() :BitVector(1) {}

	void construct_auxiliary_table() {
		//rankクエリのための補助データ構造を構築する。

		//large_blockは、65536bitごとにrank情報を持っておく。加えて、末尾には全体で立っているビットの総数を持っておく。
		large_block.clear();
		large_block.resize(((bits.size() * 64ULL + 65535ULL) / 65536ULL) + 1ULL);
		uint64_t large_pop = 0;
		for (uint64_t i = 0; i < bits.size(); i += 1024) {
			large_block[(i / 1024)] = large_pop;
			for (uint64_t j = 0; j < 1024 && i + j < bits.size(); ++j) {
				large_pop += _mm_popcnt_u64(bits[i + j]);
			}
		}
		large_block[large_block.size() - 1] = large_pop;
		assert(large_pop == popcount());

		//small_blockは、64bitごとに局所的な(=65536bit幅のブロックの内側での)rank情報を持っておく。
		//値は[0,65472]の範囲内(65472 == 65536 - 64)なので16bitで収まる。
		small_block.clear();
		small_block.resize(bits.size());

		for (uint64_t i = 0; i < bits.size(); i += 1024) {
			uint64_t pop = 0;
			for (uint64_t j = 0; j < 1024 && i + j < bits.size(); ++j) {
				assert(pop < (1ULL << 16));
				small_block[i + j] = uint16_t(pop);
				pop += _mm_popcnt_u64(bits[i + j]);
			}
		}
	}

	uint64_t rank1(const uint64_t index) {
		//bitsの範囲[0,index)のなかで立っているビットの数を返す。

		assert(index < bits.size() * 64);

		return large_block[index / 65536] + small_block[index / 64] + _mm_popcnt_u64(bits[index / 64] & ((1ULL << (index % 64)) - 1));
	}
};

bool test_bit_vector(const uint64_t seed, const int length) {
	std::mt19937_64 rnd(seed);
	std::uniform_int_distribution<uint64_t>pos_dist(100000, 200000);

	for (int i = 0; i < length; ++i) {
		if (i % (length / 10) == 0) {
			std::cout << "test_bitvector_func: " << i << " / " << length << std::endl;
		}

		const uint64_t len_vec = pos_dist(rnd);
		BitVector_nodes b(len_vec);

		for (uint64_t i = 0; i < len_vec; ++i) {
			if (rnd() % 2) {
				b.set1(i);
			}
		}

		b.construct_auxiliary_table();

		const uint64_t position = std::uniform_int_distribution<uint64_t>(0, len_vec - 1)(rnd);
		const uint64_t r1 = b.rank1(position);
		const uint64_t r2 = b.rank1_naive(position);
		if (r1 != r2) {
			std::cout << "test failed." << std::endl;
			return false;
		}

		const uint64_t s1 = b.popcount();
		const uint64_t s2 = b.rank1(len_vec);
		if (s1 != s2) {
			std::cout << "test failed." << std::endl;
			return false;
		}
	}
	std::cout << "test clear!" << std::endl;
	return true;
}

class OstleEnumerator {

protected:

	std::vector<uint64_t>all_positions;

	BitVector is_checkmate_position;

	BitVector_nodes is_nontrivial_node;

	std::vector<int16_t>all_nontrivial_solutions;

	int num_start_piece;

	constexpr static uint64_t SOLUTION_ALL_WIN = 0x5555'5555'5555'5555ULL;
	constexpr static uint64_t SOLUTION_ALL_LOSE = 0xAAAA'AAAA'AAAA'AAAAULL;

	uint64_t code2index(const uint64_t code) {
		//all_positionsのどれかの値codeを引数にとり、all_positions[i]=codeなるiを探して返す。
		//all_positionsが昇順にソートされていると仮定して二分探索で求める。

		uint64_t lower = 0, upper = all_positions.size();

		while (lower + 1ULL < upper) {
			const uint64_t mid = (lower + upper) / 2;
			if (all_positions[mid] == code)return mid;
			else if (all_positions[mid] <= code)lower = mid;
			else upper = mid;
		}

		assert(all_positions[lower] == code);
		return lower;
	}

	void output_special_positions(const uint64_t zerocount) {

		constexpr int BUFSIZE = 2 * 1024 * 1024;
		static char buf[BUFSIZE];//大きいのでスタック領域に置きたくないからstatic。（べつにmallocでもstd::vectorでもいいんだけど）

		std::ofstream writing_file;
		writing_file.rdbuf()->pubsetbuf(buf, BUFSIZE);
		writing_file.open("ostle_special_positions.txt", std::ios::out);

		std::string numcode;
		uint64_t count = 0;
		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			if (is_nontrivial_node.get(i * 25) == false && is_checkmate_position.get(i) == false) {
				++count;

				encode_25numbers(all_positions[i], numcode);
				writing_file << numcode << std::endl;
			}
		}

		writing_file.close();

		assert(count == zerocount);
	}

	uint64_t count_nontrivial_node_and_make_bitvector(const bool is_checkmate_trivial = true) {
		//all_positionsには全盤面が格納されていると仮定する。
		//全盤面と全禁じ手の有無との組み合わせによって想定される全局面のうち、
		//自明でない（＝他の盤面から合法手によって到達可能であり、対局が続いており、かつチェックメイト盤面ではない）局面の数を求めて返す。

		std::cout << "LOG: [" << get_datetime_str() << "] start: count_nontrivial_node_and_make_bitvector" << std::endl;

		//結果を得るための配列を確保
		is_nontrivial_node = BitVector_nodes(all_positions.size() * 25);
		is_checkmate_position = BitVector(all_positions.size());

		//チェックメイト盤面かどうか調べる。そうならフラグを立てておく。
		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			uint64_t bb1 = 0, bb2 = 0, pos = 0;
			decode_ostle(all_positions[i], bb1, bb2, pos);
			if (is_checkmate(bb1, bb2, pos)) {
				is_checkmate_position.set1(i);
			}
		}

		{
			const uint64_t length_itos = my_itos(all_positions.size(), 0, ' ').size();
			const double percentage = 100.0 * double(is_checkmate_position.popcount()) / double(all_positions.size());
			std::cout << "result: number of checkmate positions     = " << my_itos(is_checkmate_position.popcount(), length_itos, ' ') << " / " << all_positions.size()
				<< " (" << percentage << " %)" << std::endl;
			std::cout << "result: number of non-checkmate positions = " << my_itos(all_positions.size() - is_checkmate_position.popcount(), length_itos, ' ') << std::endl;
		}

		const int64_t siz = all_positions.size();

#pragma omp parallel for schedule(guided)
		for (int64_t i = 0; i < siz; ++i) {

			//チェックメイト盤面からチェックメイトを見逃すことで辿り着けるノードも完全解析の対象とするので、次のif文は使わない。
			//if (is_checkmate_position.get(i))continue;

			uint64_t bb_player = 0, bb_opponent = 0, pos_hole = 0;
			decode_ostle(all_positions[i], bb_player, bb_opponent, pos_hole);

			Moves moves, next_moves;

			generate_moves(bb_player, bb_opponent, pos_hole, moves);
			assert(moves[0] <= 24);

			//結果を書き出しておくための配列。これによってクリティカルセクションに出入りする頻度を減らす。
			uint64_t result[25] = {}, num_result = 0;

			//盤面all_positions[i]から指し手moves[j]によって到達するノードは、到達可能なノードである。
			for (uint8_t j = 1; j <= moves[0]; ++j) {

				uint64_t next_bb_player = bb_player, next_bb_opponent = bb_opponent, next_pos_hole = pos_hole;
				do_move(next_bb_player, next_bb_opponent, next_pos_hole, moves[j]);

				//自殺手とチェックメイトを実行する手を指した後の盤面は考えない。
				if (_mm_popcnt_u64(next_bb_player) == 3)continue;
				if (_mm_popcnt_u64(next_bb_opponent) == 3)continue;

				const uint64_t next_index = code2index(code_unique(encode_ostle(next_bb_opponent, next_bb_player, next_pos_hole)));

				if (is_checkmate_trivial) {
					//到達した先の盤面がチェックメイトの場合は除外する。
					if (is_checkmate_position.get(next_index))continue;
				}

				//到達した先の盤面から更に1手指したらもとに戻るならば禁じ手である。それを探すことで、ノードを確定させる。
				generate_moves(next_bb_opponent, next_bb_player, next_pos_hole, next_moves);
				uint64_t next_forbidden_index = 0;
				for (uint8_t k = 1; k <= next_moves[0]; ++k) {
					uint64_t bb1 = next_bb_opponent, bb2 = next_bb_player, pos = next_pos_hole;
					do_move(bb1, bb2, pos, next_moves[k]);
					if (bb2 == bb_player && bb1 == bb_opponent && pos == pos_hole) {
						next_forbidden_index = k;
						break;
					}
				}

				result[num_result++] = next_index * 25ULL + next_forbidden_index;
			}

#pragma omp critical
			{
				for (uint64_t j = 0; j < num_result; ++j) {
					is_nontrivial_node.set1(result[j]);
				}
			}
		}

		is_nontrivial_node.construct_auxiliary_table();

		{
			const double percentage = 100.0 * double(is_nontrivial_node.popcount()) / double(all_positions.size() * 25);
			std::cout << "result: number of non-trivial nodes = " << is_nontrivial_node.popcount() << " / " << all_positions.size() * 25
				<< " (" << percentage << " %)" << std::endl;
		}

		{
			uint64_t zerocount = 0;
			for (uint64_t i = 0; i < all_positions.size(); ++i) {
				if (is_nontrivial_node.get(i * 25) == false && is_checkmate_position.get(i) == false)++zerocount;
			}
			std::cout << "result: number of nodes s.t. non-checkmate & no forbidden move & trivial(unreachable) = " << zerocount << " / " << all_positions.size()
				<< " (" << (100.0 * double(zerocount) / double(all_positions.size())) << " %)" << std::endl;

			const uint64_t numer = is_nontrivial_node.popcount() - (all_positions.size() - is_checkmate_position.popcount() - zerocount), denom = is_nontrivial_node.popcount();
			std::cout << "result: number of non-trivial nodes with forbidden move = "
				<< numer << " / " << denom << " (" << (100.0 * double(numer) / double(denom)) << " %)" << std::endl;

			if (zerocount > 0) {
				output_special_positions(zerocount);
			}
		}

		std::cout << "LOG: [" << get_datetime_str() << "] finish: count_nontrivial_node_and_make_bitvector" << std::endl;

		return is_nontrivial_node.popcount();
	}

public:

	OstleEnumerator(const int start_piece) {

		if (start_piece == 10) {
			all_positions.reserve(2'800'000'000ULL);
		}
		else if (start_piece == 9) {
			all_positions.reserve(1'200'000'000ULL);
		}
		else if (start_piece == 8) {
			all_positions.reserve(200'000'000ULL);
		}
		else {
			assert(false);
		}
		num_start_piece = start_piece;
	}

	OstleEnumerator() = delete;

	void do_enumerate() {
		PositionEnumerator p;
		p.do_enumerate(num_start_piece, all_positions);
	}

	uint64_t calc_final_result_hashvalue() {
		//全盤面とその解析結果の組に関するハッシュ値を生成して返す。

		Encoder_AES encoder(123456);

		__m128i answer = _mm_setzero_si128();

		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			const __m128i x = encoder.aes128_enc(all_positions[i], i);
			answer = _mm_xor_si128(x, answer);//xorは可換
		}

		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			const __m128i x = encoder.aes128_enc(is_nontrivial_node.get_bb_position(i), i);
			answer = _mm_xor_si128(x, answer);//xorは可換
		}

		for (uint64_t i = 0; i < all_nontrivial_solutions.size(); ++i) {
			const __m128i x = encoder.aes128_enc(all_nontrivial_solutions[i], i);
			answer = _mm_xor_si128(x, answer);//xorは可換
		}

		uint64_t a[2];
		_mm_storeu_si128((__m128i*)a, answer);
		return a[0] ^ a[1];
	}

	void print_statistics_of_results() {
		std::cout << "LOG: [" << get_datetime_str() << "] start: print_statistics_of_results" << std::endl;
		uint64_t max_num = 0;
		for (uint64_t i = 0; i < all_nontrivial_solutions.size(); ++i) {
			max_num = std::max(max_num, uint64_t(all_nontrivial_solutions[i]));
		}
		std::vector<uint64_t>histogram(max_num + 1, 0);
		for (uint64_t i = 0; i < all_nontrivial_solutions.size(); ++i) {
			++histogram[all_nontrivial_solutions[i]];
		}
		std::cout << "result: number of nontrivial nodes of which distance to endgame is:" << std::endl;
		for (uint64_t i = 0; i < histogram.size(); ++i) {
			std::cout << "result: " << i << " : " << histogram[i] << std::endl;
		}
		std::cout << "LOG: [" << get_datetime_str() << "] finish: print_statistics_of_results" << std::endl;
	}

	void output_positions_and_solutions(const std::string filename) {

		std::cout << "LOG: [" << get_datetime_str() << "] start: output_positions_and_solutions" << std::endl;

		constexpr int BUFSIZE = 2 * 1024 * 1024;
		static char buf[BUFSIZE];//大きいのでスタック領域に置きたくないからstatic。（べつにmallocでもstd::vectorでもいいんだけど）

		std::ofstream writing_file;
		writing_file.rdbuf()->pubsetbuf(buf, BUFSIZE);
		uint64_t count = 0;
		std::string text;
		constexpr uint64_t SINGLE_FILE_LIMIT = 1'000'000;
		const std::string SERIAL_TRIVIAL_CODE = "-ABCDEFGHIJKLMNOPQRSTUVWXYZ";

		uint64_t node_index = 0;
		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			if (count % SINGLE_FILE_LIMIT == 0) {
				if (count) {
					writing_file.close();
				}
				writing_file.open(filename + my_itos(count / SINGLE_FILE_LIMIT, 4, '0') + std::string(".txt"), std::ios::out);
			}

			encode_base64(all_positions[i], text);
			const uint64_t bb = is_nontrivial_node.get_bb_position(i);
			if (is_checkmate_position.get(i)) {
				assert(bb == 0);
				text += ",checkmate";
			}
			else {
				for (uint64_t j = 0; j < 25; ++j) {
					text += ",";
					if (bb & (1ULL << j)) {
						text += std::to_string(all_nontrivial_solutions[node_index++]);
					}
				}
			}

			writing_file << text << std::endl;
			++count;
		}
		if (all_positions.size()) {
			writing_file.close();
		}

		std::cout << "LOG: [" << get_datetime_str() << "] finish: output_positions_and_solutions" << std::endl;

	}
};

class OstleRetrogradeAnalyzer : public OstleEnumerator {

private:

	uint64_t check_one_position(const uint64_t index, std::vector<uint16_t> &dest) {
		//この関数が呼ばれる前にcount_nontrivial_node_and_make_bitvector関数が呼ばれていることを仮定する。
		//盤面all_positions[index]の全てのノード（禁じ手によって区別される）について勝敗判定を試みる。
		//返り値は、今回の試行で新たに勝敗判定がなされたなら、新たになされた数を返す。さもなくば0を返す。

		dest.clear();

		//盤面all_positions[index]の全てのノードが自明であるならば0を返す。さもなくば、非自明なノード（たち）の現在の情報をdestにコピーする。
		{
			const uint64_t is_nontrivial_bb = is_nontrivial_node.get_bb_position(index);
			if (is_nontrivial_bb == 0) {
				return 0;
			}
			const uint64_t start_rank = is_nontrivial_node.rank1(index * 25);
			const uint64_t pop = _mm_popcnt_u64(is_nontrivial_bb);
			for (uint64_t i = 0; i < pop; ++i) {
				dest.push_back(all_nontrivial_solutions[start_rank + i]);
			}
		}

		//処理がここに到達したということは、
		//盤面all_positions[index]が内包するノードのうちには非自明なノードが存在する。

		uint64_t bb_player = 0, bb_opponent = 0, pos_hole = 0;
		decode_ostle(all_positions[index], bb_player, bb_opponent, pos_hole);

		Moves moves, next_moves;
		generate_moves(bb_player, bb_opponent, pos_hole, moves);

		assert(moves[0] <= 24);

		//まず、各指し手を指した先の局面が勝敗判定されているか調べる。
		uint64_t outcome[32] = {}, win_num = 0, lose_num = 0, unknown_num = 0, suicide_num = 0;
		int is_suicide[32] = {};
		for (uint8_t i = 1; i <= moves[0]; ++i) {
			uint64_t next_bb_player = bb_player, next_bb_opponent = bb_opponent, next_pos_hole = pos_hole;
			do_move(next_bb_player, next_bb_opponent, next_pos_hole, moves[i]);
			if (_mm_popcnt_u64(next_bb_player) == 3) {
				is_suicide[i] = 1;
				++suicide_num;
				continue;
			}

			const uint64_t next_position_index = code2index(code_unique(encode_ostle(next_bb_opponent, next_bb_player, next_pos_hole)));

			//指すとチェックメイト盤面になる手は、指すと負ける手である。
			if (is_checkmate_position.get(next_position_index)) {
				outcome[i] = 1;
				++lose_num;
				continue;
			}

			//到達した先の盤面から更に1手指したらもとに戻るならば禁じ手である。それを探すことで、どのノードに到達したのか求める。
			generate_moves(next_bb_opponent, next_bb_player, next_pos_hole, next_moves);
			int forbidden_index = 0;
			for (uint8_t j = 1; j <= next_moves[0]; ++j) {
				uint64_t bb1 = next_bb_opponent, bb2 = next_bb_player, pos = next_pos_hole;
				do_move(bb1, bb2, pos, next_moves[j]);
				if (bb2 == bb_player && bb1 == bb_opponent && pos == pos_hole) {
					forbidden_index = j;
					break;
				}
			}
			const uint64_t next_node_index = next_position_index * 25 + forbidden_index;
			const uint64_t next_node_rank = is_nontrivial_node.rank1(next_node_index);
			assert(is_nontrivial_node.get(next_node_index));

			const uint64_t next_solution = all_nontrivial_solutions[next_node_rank];
			if (next_solution == 0) {
				outcome[i] = 0;
				++unknown_num;
			}
			else if (next_solution % 2 == 1) {
				//i番目の指し手を指した先の局面から、奇数の手数でチェックメイト盤面に到達するということは、勝てる指し手だということ
				outcome[i] = next_solution + 1;//勝てる指し手のoutcomeは偶数
				++win_num;
			}
			else {
				//i番目の指し手を指した先の局面から、偶数の手数でチェックメイト盤面に到達するということは、負ける指し手だということ
				outcome[i] = next_solution + 1;//負ける指し手のoutcomeは奇数
				++lose_num;
			}
		}

		assert(int(win_num) + int(lose_num) + int(unknown_num) + int(suicide_num) == int(moves[0]));

		uint64_t answer = 0;

		//2つ以上の指し手が勝利をもたらすなら、禁じ手がどれであろうと勝利が確定する。
		if (win_num >= 2) {

			//最短で勝利できる手（最短勝利手）の添字と、最短勝利手が複数あるかどうかを調べる。
			int min_index = -1;
			bool tie_flag = false;
			for (int i = 1; i <= moves[0]; ++i) {
				if (outcome[i] != 0 && outcome[i] % 2 == 0) {
					if (min_index == -1) {
						min_index = i;
					}
					else {
						if (outcome[i] < outcome[min_index]) {
							min_index = i;
							tie_flag = false;
						}
						else if (outcome[i] == outcome[min_index]) {
							tie_flag = true;
						}
					}
				}
			}

			uint64_t is_nontrivial_bb = is_nontrivial_node.get_bb_position(index);
			for (uint32_t i = 0, x = 0; bitscan_forward64(is_nontrivial_bb, &x); ++i, is_nontrivial_bb &= is_nontrivial_bb - 1) {

				//「そのノードに禁じ手が無いか、そのノードの禁じ手が最短勝利手ではないか、最短勝利手が複数ある」ならば、そのノードは最短勝利手の手数で勝利できる。
				if (x != min_index || tie_flag) {
					assert(min_index != -1);
					if (dest[i] != outcome[min_index]) {
						++answer;
						dest[i] = outcome[min_index];
					}
				}
				else {
					//唯一の最短勝利手が禁じ手のノードは、二番目に速く勝利できる手の手数で勝利できる。
					int second_min_index = -1;
					for (int j = 1; j <= moves[0]; ++j) {
						if (outcome[j] != 0 && outcome[j] % 2 == 0 && outcome[j] != outcome[min_index]) {
							if (second_min_index == -1) {
								second_min_index = j;
							}
							else {
								if (outcome[j] < outcome[second_min_index]) {
									second_min_index = j;
								}
							}
						}
					}
					assert(second_min_index != -1);
					if (dest[i] != outcome[second_min_index]) {
						++answer;
						dest[i] = outcome[second_min_index];
					}
				}
			}

			return answer;
		}

		//1つの指し手Xだけが勝利確定手の場合
		if (win_num == 1) {

			//Xの添字を調べる。
			int win_index = -1;
			for (int i = 1; i <= moves[0]; ++i) {
				if (outcome[i] != 0 && outcome[i] % 2 == 0) {
					win_index = i;
					break;
				}
			}

			uint64_t is_nontrivial_bb = is_nontrivial_node.get_bb_position(index);
			for (uint32_t i = 0, x = 0; bitscan_forward64(is_nontrivial_bb, &x); ++i, is_nontrivial_bb &= is_nontrivial_bb - 1) {

				//「そのノードに禁じ手が無いか、そのノードの禁じ手がXではない」ならば、そのノードはXの手数で勝利できる。
				if (x != win_index) {
					assert(win_index != -1);
					if (dest[i] != outcome[win_index]) {
						++answer;
						dest[i] = outcome[win_index];
					}
				}
				else {
					//Xが禁じ手のノードについて

					//X以外が全て敗北確定の場合、そのノードの敗北が確定する。
					if (unknown_num == 0) {

						//敗北する手のうち手数が最長の手の添字を調べる。
						int max_index = -1;
						for (int j = 1; j <= moves[0]; ++j) {
							if (outcome[j] != 0 && outcome[j] % 2 == 1) {
								if (max_index == -1) {
									max_index = j;
								}
								else {
									if (outcome[j] > outcome[max_index]) {
										max_index = j;
									}
								}
							}
						}
						assert(max_index != -1);//X以外全て自殺手であるケースがもしあればここで落ちる。（ないだろうと思う。証明したければ全盤面そうでないと確かめればいい）
						if (dest[i] != outcome[max_index]) {
							++answer;
							dest[i] = outcome[max_index];
						}

					}
					else {
						//未確定の手が存在するならば、そのノードは未確定である。
						assert(dest[i] == 0);
					}
				}
			}

			return answer;
		}

		//全ての指し手が敗北確定なら、禁じ手がどれであろうと敗北が確定する。
		if (win_num == 0 && unknown_num == 0) {

			//最長手数で敗北する手（最長敗北手）の添字と、最長敗北手が複数あるかどうかを調べる。
			int max_index = -1;
			bool tie_flag = false;
			for (int i = 1; i <= moves[0]; ++i) {
				if (outcome[i] != 0 && outcome[i] % 2 == 1) {
					if (max_index == -1) {
						max_index = i;
					}
					else {
						if (outcome[i] > outcome[max_index]) {
							max_index = i;
							tie_flag = false;
						}
						else if (outcome[i] == outcome[max_index]) {
							tie_flag = true;
						}
					}
				}
			}

			uint64_t is_nontrivial_bb = is_nontrivial_node.get_bb_position(index);
			for (uint32_t i = 0, x = 0; bitscan_forward64(is_nontrivial_bb, &x); ++i, is_nontrivial_bb &= is_nontrivial_bb - 1) {

				//「そのノードに禁じ手が無いか、そのノードの禁じ手が最長敗北手ではないか、最長敗北手が複数ある」ならば、そのノードは最長敗北手の手数で敗北する。
				if (x != max_index || tie_flag) {
					assert(max_index != -1);
					if (dest[i] != outcome[max_index]) {
						++answer;
						dest[i] = outcome[max_index];
					}
				}
				else {
					//唯一の最長敗北手が禁じ手のノードは、二番目に長い手数で敗北する手の手数で敗北する。
					int second_max_index = -1;
					for (int j = 1; j <= moves[0]; ++j) {
						if (outcome[j] != 0 && outcome[j] % 2 == 1 && outcome[j] != outcome[max_index]) {
							if (second_max_index == -1) {
								second_max_index = j;
							}
							else {
								if (outcome[j] > outcome[second_max_index]) {
									second_max_index = j;
								}
							}
						}
					}
					assert(second_max_index != -1);//唯一の最長敗北手以外全て自殺手であるケースがもしあればここで落ちる。（ないだろうと思う。証明したければ全盤面そうでないと確かめればいい）
					if (dest[i] != outcome[second_max_index]) {
						++answer;
						dest[i] = outcome[second_max_index];
					}
				}
			}

			return answer;
		}

		//ただ1つの指し手Xが未確定で、それ以外全ての指し手が敗北確定の場合
		if (win_num == 0 && unknown_num == 1) {

			//Xの添字を調べる。
			int unknown_index = -1;
			for (int i = 1; i <= moves[0]; ++i) {
				if (is_suicide[i] == 0 && outcome[i] == 0) {
					unknown_index = i;
					break;
				}
			}

			uint64_t is_nontrivial_bb = is_nontrivial_node.get_bb_position(index);
			for (uint32_t i = 0, x = 0; bitscan_forward64(is_nontrivial_bb, &x); ++i, is_nontrivial_bb &= is_nontrivial_bb - 1) {

				//「そのノードに禁じ手が無いか、そのノードの禁じ手がXではない」ならば、そのノードは未確定である。
				if (x != unknown_index) {
					assert(unknown_index != -1);
					assert(dest[i] == 0);
				}
				else {
					//Xが禁じ手のノードは敗北が確定する。

					//敗北する手のうち手数が最長の手の添字を調べる。
					int max_index = -1;
					for (int j = 1; j <= moves[0]; ++j) {
						if (outcome[j] != 0 && outcome[j] % 2 == 1) {
							if (max_index == -1) {
								max_index = j;
							}
							else {
								if (outcome[j] > outcome[max_index]) {
									max_index = j;
								}
							}
						}
					}
					assert(max_index != -1);//X以外全て自殺手であるケースがもしあればここで落ちる。（ないだろうと思う。証明したければ全盤面そうでないと確かめればいい）
					if (dest[i] != outcome[max_index]) {
						++answer;
						dest[i] = outcome[max_index];
					}
				}
			}

			return answer;
		}

		//ここに到達したということは、勝利確定手が見つかっておらず、かつ未確定の指し手が2つ以上存在する。
		assert(win_num == 0 && unknown_num >= 2);

		//その場合は禁じ手がどれであろうと未確定である。

		uint64_t is_nontrivial_bb = is_nontrivial_node.get_bb_position(index);
		for (uint32_t i = 0, x = 0; bitscan_forward64(is_nontrivial_bb, &x); ++i, is_nontrivial_bb &= is_nontrivial_bb - 1) {
			assert(dest[i] == 0);
		}

		return 0;
	}

	std::pair<uint64_t, uint64_t> retrograde_analysis_single_iteration() {

		const uint64_t TASK_SIZE = 1'000'000ULL;
		const int64_t CHANK_SIZE = 100;

		std::vector<uint64_t>next_solutions;
		next_solutions.reserve(TASK_SIZE);

		uint64_t updated_num = 0, denovo = 0;

		for (uint64_t t = 0; t < all_positions.size(); t += TASK_SIZE) {

			const int64_t start = t, end = std::min(t + TASK_SIZE, uint64_t(all_positions.size()));

			const uint64_t offset = is_nontrivial_node.rank1(start * 25);
			next_solutions.resize(is_nontrivial_node.rank1(end * 25) - offset);

#pragma omp parallel for schedule(guided)
			for (int64_t i = start; i < end; i += CHANK_SIZE) {
				std::vector<uint16_t>dest;
				dest.reserve(25);

				for (int64_t j = i; j < i + CHANK_SIZE && j < end; j++) {
					check_one_position(j, dest);
					const uint64_t start_rank = is_nontrivial_node.rank1(uint64_t(j * 25));
					for (uint64_t k = 0; k < dest.size(); ++k) {
						next_solutions[start_rank - offset + k] = dest[k];
					}
				}
			}

			for (uint64_t i = 0; i < next_solutions.size(); ++i) {
				if (all_nontrivial_solutions[offset + i] != next_solutions[i]) {
					++updated_num;
					if (all_nontrivial_solutions[offset + i] == 0) {
						++denovo;
					}
					else {
						assert(next_solutions[i] != 0);
						assert((all_nontrivial_solutions[offset + i] % 2) == (next_solutions[i] % 2));
					}
					all_nontrivial_solutions[offset + i] = next_solutions[i];
				}
			}
		}
		return { updated_num, denovo };
	}

public:

	OstleRetrogradeAnalyzer(const int start_piece) : OstleEnumerator(start_piece) {}

	OstleRetrogradeAnalyzer() = delete;

	void retrograde_analysis() {

		do_enumerate();

		std::cout << "LOG: [" << get_datetime_str() << "] start: retrograde_analysis" << std::endl;

		count_nontrivial_node_and_make_bitvector();

		all_nontrivial_solutions.clear();
		all_nontrivial_solutions.resize(is_nontrivial_node.popcount());

		for (int iteration = 1;; ++iteration) {
			std::cout << "LOG: [" << get_datetime_str() << "] start: iteration " << iteration << std::endl;
			const auto t = std::chrono::system_clock::now();

			const auto updated_num = retrograde_analysis_single_iteration();

			const auto s = std::chrono::system_clock::now();
			const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
			std::cout << "LOG: [" << get_datetime_str() << "] finish: iteration " << iteration
				<< " : updated_num = " << updated_num.first << "(denovo = " << updated_num.second << "), elapsed time = " << elapsed << " ms" << std::endl;
			if (updated_num.first == 0)break;
		}

		std::cout << "LOG: [" << get_datetime_str() << "] finish: retrograde_analysis" << std::endl;
	}

	int query(
		const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole,
		const uint64_t forbidden_bb_player, const uint64_t forbidden_bb_opponent, const uint64_t forbidden_pos_hole) {

		const bool valid_input =
			((bb_player & BB_ALL_8X8_5X5) == bb_player) &&
			((bb_opponent & BB_ALL_8X8_5X5) == bb_opponent) &&
			((bb_player & bb_opponent) == 0) &&
			(pos_hole < 25) &&
			(4 <= _mm_popcnt_u64(bb_player) && _mm_popcnt_u64(bb_player) <= 5) &&
			(4 <= _mm_popcnt_u64(bb_opponent) && _mm_popcnt_u64(bb_opponent) <= 5);

		if (!valid_input)return -1;//invalidな盤面が入力された

		const auto c2i_or_error = [&](const uint64_t c) {
			uint64_t lower = 0, upper = all_positions.size();

			while (lower + 1ULL < upper) {
				const uint64_t mid = (lower + upper) / 2;
				if (all_positions[mid] == c)return mid;
				else if (all_positions[mid] <= c)lower = mid;
				else upper = mid;
			}

			if (all_positions[lower] != c)return uint64_t(0xFFFF'FFFF'FFFF'FFFFULL);//error

			return lower;
		};

		const uint64_t index = c2i_or_error(code_unique(encode_ostle(bb_player, bb_opponent, pos_hole)));
		if (index == 0xFFFF'FFFF'FFFF'FFFFULL)return -2;//なぜかall_positions配列のなかにに入力盤面が存在しなかった

		if (is_checkmate(bb_player, bb_opponent, pos_hole))return -3;//チェックメイト盤面である

		Moves moves;

		generate_moves(bb_player, bb_opponent, pos_hole, moves);
		uint64_t forbidden_index = 0;
		for (uint8_t i = 1; i <= moves[0]; ++i) {
			uint64_t bb1 = bb_player, bb2 = bb_opponent, pos = pos_hole;
			do_move(bb1, bb2, pos, moves[i]);
			if (bb1 == forbidden_bb_player && bb2 == forbidden_bb_opponent && pos == forbidden_pos_hole) {
				forbidden_index = i;
				break;
			}
		}

		if (!is_nontrivial_node.get(index * 25 + forbidden_index))return -4;//入力の（盤面、禁じ手）の組み合わせに到達できない

		const uint64_t rank = is_nontrivial_node.rank1(index * 25 + forbidden_index);

		if (all_nontrivial_solutions.size() <= rank)return -5;//なぜかall_nontrivial_solutions配列のなかに入力局面が存在しなかった

		return all_nontrivial_solutions[rank];//返り値が0ならば引き分け、正の数ならばチェックメイトまでの手数（奇数なら手番側の勝ち、偶数なら手番側の負け）
	}

	void answer_about_initial_position() {
		std::cout << "LOG: [" << get_datetime_str() << "] start: answer_about_initial_position" << std::endl;

		const uint64_t bb_player = 0b11111ULL << 32;
		const uint64_t bb_opponent = 0b11111ULL;

		visualize_ostle(bb_player, bb_opponent, 12);
		const int answer = query(bb_player, bb_opponent, 12, 0, 0, 0);

		if (answer == 0) {
			std::cout << "result: the initial position is draw." << std::endl;
		}
		else if (answer < 0) {
			std::cout << "result: query error:" << answer << std::endl;
		}
		else if (answer % 2 == 0) {
			std::cout << "result: player wins in the initial position with " << answer << " moves." << std::endl;
		}
		else {
			std::cout << "result: opponent wins in the initial position with " << answer << " moves." << std::endl;
		}

		std::cout << "LOG: [" << get_datetime_str() << "] finish: answer_about_initial_position" << std::endl;
	}



};

class OstleBreadthFirstSearcher : public OstleEnumerator {

private:

	BitVector now_nodelist, next_nodelist;

	uint64_t bfs_single_iteration(const uint64_t next_depth) {

		assert(next_nodelist.popcount() == 0);

		const int64_t siz1 = int64_t(is_nontrivial_node.get_size());

#pragma omp parallel for schedule(dynamic, 65536)
		for (int64_t i = 0; i < siz1; ++i) {
			if (!is_nontrivial_node.get(i))continue;
			const uint64_t node_rank = is_nontrivial_node.rank1(i);
			if (!now_nodelist.get(node_rank))continue;
			const uint64_t now_code = all_positions[i / 25];
			const uint64_t now_forbbiden_index = i % 25;

			uint64_t bb_player = 0, bb_opponent = 0, pos_hole = 0;
			decode_ostle(now_code, bb_player, bb_opponent, pos_hole);
			Moves moves, next_moves;
			generate_moves(bb_player, bb_opponent, pos_hole, moves);



			assert(moves[0] <= 24);

			//結果を書き出しておくための配列。これによってクリティカルセクションに出入りする頻度を減らす。
			uint64_t result[25] = {}, num_result = 0;

			//盤面all_positions[i]から指し手moves[j]によって到達するノードは、到達可能なノードである。
			for (uint8_t j = 1; j <= moves[0]; ++j) {

				uint64_t next_bb_player = bb_player, next_bb_opponent = bb_opponent, next_pos_hole = pos_hole;
				do_move(next_bb_player, next_bb_opponent, next_pos_hole, moves[j]);

				//自殺手とチェックメイトを実行する手を指した後の盤面は考えない。
				if (_mm_popcnt_u64(next_bb_player) == 3)continue;
				if (_mm_popcnt_u64(next_bb_opponent) == 3)continue;

				const uint64_t next_index = code2index(code_unique(encode_ostle(next_bb_opponent, next_bb_player, next_pos_hole)));

				//到達した先の盤面から更に1手指したらもとに戻るならば禁じ手である。それを探すことで、ノードを確定させる。
				generate_moves(next_bb_opponent, next_bb_player, next_pos_hole, next_moves);
				uint64_t next_forbidden_index = 0;
				for (uint8_t k = 1; k <= next_moves[0]; ++k) {
					uint64_t bb1 = next_bb_opponent, bb2 = next_bb_player, pos = next_pos_hole;
					do_move(bb1, bb2, pos, next_moves[k]);
					if (bb2 == bb_player && bb1 == bb_opponent && pos == pos_hole) {
						next_forbidden_index = k;
						break;
					}
				}

				result[num_result++] = is_nontrivial_node.rank1(next_index * 25ULL + next_forbidden_index);
			}

#pragma omp critical
			{
				for (uint64_t j = 0; j < num_result; ++j) {
					next_nodelist.set1(result[j]);
				}
			}
		}

		const int64_t siz2 = int64_t(next_nodelist.get_size());

		std::cout << "LOG: [" << get_datetime_str() << "] mid: bfs" << std::endl;

		now_nodelist.set0_all();

		uint64_t answer = 0;
		for (int64_t i = 0; i < siz2; ++i) {
			if (next_nodelist.get(i)) {
				if (all_nontrivial_solutions[i] == -1) {
					all_nontrivial_solutions[i] = next_depth;
					now_nodelist.set1(i);
				}
			}
		}

		next_nodelist.set0_all();

		return now_nodelist.popcount();
	}

public:

	OstleBreadthFirstSearcher(const int start_piece) : OstleEnumerator(start_piece) {}

	OstleBreadthFirstSearcher() = delete;

	void bfs() {

		do_enumerate();

		std::cout << "LOG: [" << get_datetime_str() << "] start: bfs" << std::endl;

		count_nontrivial_node_and_make_bitvector(false);

		now_nodelist = BitVector(is_nontrivial_node.popcount());
		next_nodelist = BitVector(is_nontrivial_node.popcount());

		all_nontrivial_solutions.clear();
		all_nontrivial_solutions.resize(is_nontrivial_node.popcount(), -1);

		const uint64_t initial_position_code = num_start_piece == 10 ?
			encode_ostle(0b11111ULL << 32, 0b11111ULL, 12) : (num_start_piece == 9 ?
				encode_ostle(0b11111ULL << 32, 0b01111ULL, 12) :
				encode_ostle(0b01111ULL << 32, 0b01111ULL, 12));
		const uint64_t initial_position_index = code2index(code_unique(initial_position_code));
		const uint64_t initial_node_index = initial_position_index * 25;
		const uint64_t initial_node_rank = is_nontrivial_node.rank1(initial_node_index);
		now_nodelist.set1(initial_node_rank);
		all_nontrivial_solutions[initial_node_rank] = 0;

		for (int iteration = 1;; ++iteration) {
			std::cout << "LOG: [" << get_datetime_str() << "] start: iteration " << iteration << std::endl;
			const auto t = std::chrono::system_clock::now();

			const uint64_t updated_num = bfs_single_iteration(iteration);

			const auto s = std::chrono::system_clock::now();
			const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
			std::cout << "LOG: [" << get_datetime_str() << "] finish: iteration " << iteration
				<< " : updated_num = " << updated_num << ", elapsed time = " << elapsed << " ms" << std::endl;
			if (updated_num == 0)break;
		}

		std::cout << "LOG: [" << get_datetime_str() << "] finish: bfs" << std::endl;
	}

	uint64_t calc_final_result_hashvalue() {
		//全盤面とその解析結果の組に関するハッシュ値を生成して返す。

		Encoder_AES encoder(123456);

		__m128i answer = _mm_setzero_si128();

		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			const __m128i x = encoder.aes128_enc(all_positions[i], i);
			answer = _mm_xor_si128(x, answer);//xorは可換
		}

		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			const __m128i x = encoder.aes128_enc(is_nontrivial_node.get_bb_position(i), i);
			answer = _mm_xor_si128(x, answer);//xorは可換
		}

		for (uint64_t i = 0; i < all_nontrivial_solutions.size(); ++i) {
			const __m128i x = encoder.aes128_enc(all_nontrivial_solutions[i], i);
			answer = _mm_xor_si128(x, answer);//xorは可換
		}

		uint64_t a[2];
		_mm_storeu_si128((__m128i*)a, answer);
		return a[0] ^ a[1];
	}

	void print_statistics_of_results() {
		std::cout << "LOG: [" << get_datetime_str() << "] start: print_statistics_of_results" << std::endl;
		uint64_t max_num = 0;
		for (uint64_t i = 0; i < all_nontrivial_solutions.size(); ++i) {
			max_num = std::max(max_num, uint64_t(all_nontrivial_solutions[i]));
		}
		std::vector<uint64_t>histogram(max_num + 1, 0);
		for (uint64_t i = 0; i < all_nontrivial_solutions.size(); ++i) {
			++histogram[all_nontrivial_solutions[i]];
		}
		std::cout << "result: number of nontrivial nodes of which distance to endgame is:" << std::endl;
		for (uint64_t i = 0; i < histogram.size(); ++i) {
			std::cout << "result: " << i << " : " << histogram[i] << std::endl;
		}
		std::cout << "LOG: [" << get_datetime_str() << "] finish: print_statistics_of_results" << std::endl;
	}

	void output_positions_and_solutions(const std::string filename) {

		std::cout << "LOG: [" << get_datetime_str() << "] start: output_positions_and_solutions" << std::endl;

		constexpr int BUFSIZE = 2 * 1024 * 1024;
		static char buf[BUFSIZE];//大きいのでスタック領域に置きたくないからstatic。（べつにmallocでもstd::vectorでもいいんだけど）

		std::ofstream writing_file;
		writing_file.rdbuf()->pubsetbuf(buf, BUFSIZE);
		uint64_t count = 0;
		std::string text;
		constexpr uint64_t SINGLE_FILE_LIMIT = 1'000'000;
		const std::string SERIAL_TRIVIAL_CODE = "-ABCDEFGHIJKLMNOPQRSTUVWXYZ";

		uint64_t node_index = 0;
		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			if (count % SINGLE_FILE_LIMIT == 0) {
				if (count) {
					writing_file.close();
				}
				writing_file.open(filename + my_itos(count / SINGLE_FILE_LIMIT, 4, '0') + std::string(".txt"), std::ios::out);
			}

			encode_base64(all_positions[i], text);
			const uint64_t bb = is_nontrivial_node.get_bb_position(i);

			if (is_checkmate_position.get(i)) {
				text += ",checkmate";
			}
			for (uint64_t j = 0; j < 25; ++j) {
				text += ",";
				if (bb & (1ULL << j)) {
					text += std::to_string(all_nontrivial_solutions[node_index++]);
				}
			}

			writing_file << text << std::endl;
			++count;
		}
		if (all_positions.size()) {
			writing_file.close();
		}

		std::cout << "LOG: [" << get_datetime_str() << "] finish: output_positions_and_solutions" << std::endl;

	}


};

uint64_t solve_8() {
	std::cout << "LOG: [" << get_datetime_str() << "] start: solve_8" << std::endl;
	OstleRetrogradeAnalyzer e(8);
	e.retrograde_analysis();
	const uint64_t fingerprint = e.calc_final_result_hashvalue();
	e.print_statistics_of_results();
	e.output_positions_and_solutions("ostle_output");
	std::cout << "LOG: [" << get_datetime_str() << "] finish: solve_8. fingerprint = " << fingerprint << std::endl;
	return fingerprint;
}


uint64_t solve_8_bfs() {
	std::cout << "LOG: [" << get_datetime_str() << "] start: solve_8_bfs" << std::endl;
	OstleBreadthFirstSearcher e(8);
	e.bfs();
	const uint64_t fingerprint = e.calc_final_result_hashvalue();
	e.print_statistics_of_results();
	e.output_positions_and_solutions("ostle_8_distance");
	std::cout << "LOG: [" << get_datetime_str() << "] finish: solve_8_bfs. fingerprint = " << fingerprint << std::endl;
	return fingerprint;
}



void unittests() {

	test_move(12345, 100000);
	test_checkmate_detector_func(12345, 100000);
	test_bitboard_symmetry(12345, 100000);
	test_base64_func(12345, 100000);
	test_bit_vector(12345, 100000);

	std::cout << "test clear!" << std::endl;
}



int main(int argc, char *argv[]) {

	//note:
	//uint64_tがなにかのエイリアスであることは規定されているが、unsigned long longのエイリアスであるとは限らない。
	//処理系によってはlongが64bitでuint64_tがunsigned longのエイリアスであることもある。（言語仕様で許容されている）
	//整数リテラルの末尾にULLを付けるとunsigned long long型であることが明示される。uint64_tではなくunsigned long longになることが問題である。
	//例えばstd::max関数は2つの引数が同じ型でなければならないのだが、uint64_tとunsigned long longを入れたときにコンパイルエラーになる可能性が処理系によってありうる。

#ifdef _MSC_VER
	SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
#endif

	std::vector<std::string>args;
	for (int i = 1; i < argc; ++i) {
		args.push_back(argv[i]);
	}

	std::map<std::string, std::string>input;
	std::string operand = "";
	for (uint64_t i = 0; i < args.size(); ++i) {
		if (operand == "") {
			if (args[i] == "-r" || args[i] == "--retrograde") {
				input["retrograde"] = "true";
			}
			else if (args[i] == "-t" || args[i] == "--test") {
				input["test"] = "true";
			}
			else if (args[i] == "-b" || args[i] == "bfs") {
				input["bfs"] = "true";
			}
			else if (args[i] == "-p" || args[i] == "--parallel") {
				operand = "parallel";
			}
			else if (args[i] == "-s" || args[i] == "--startpiece") {
				operand = "startpiece";
			}
			else {
				std::cout << "error: command line argument is invalid. error_code = 1 (cf. main function of the source code)" << std::endl;
				return 0;
			}
		}
		else if (operand == "parallel") {
			if (std::regex_match(args[i], std::regex(R"(0*[1-9][0-9]{,5})"))) {
				operand = "";
				const int par = std::stoi(args[i]);
				assert(1 <= par && par <= 10000000);
				input["parallel"] = std::to_string(par);
			}
			else if (args[i] == "-1") {
				input["parallel"] = "-1";
			}
			else {
				std::cout << "error: command line argument is invalid. error_code = 2 (cf. main function of the source code)" << std::endl;
				return 0;
			}
		}
		else if (operand == "startpiece") {
			if (args[i] == std::string("8") || args[i] == std::string("9") || args[i] == std::string("10")) {
				operand = "";
				input["startpiece"] = args[i];
			}
			else {
				std::cout << "error: command line argument is invalid. error_code = 3 (cf. main function of the source code)" << std::endl;
				return 0;
			}
		}
		else {
			std::cout << "error: command line argument is invalid. error_code = 4 (cf. main function of the source code)" << std::endl;
			return 0;
		}
	}
	if (operand != std::string("")) {
		std::cout << "error: command line argument is invalid. error_code = 5 (cf. main function of the source code)" << std::endl;
		return 0;
	}

	const int num_logical_core = omp_get_num_threads();
	if (input.find("parallel") == input.end()) {
		omp_set_num_threads(1);
	}
	else {
		const int par = std::stoi(input["parallel"]);
		if (1 <= par && par < omp_get_max_threads()) {
			omp_set_num_threads(par);
		}
		else {
			std::cout << "warning: parallelism setting is ignored. input = " << par << ", omp_get_max_threads() = " << omp_get_max_threads() << std::endl;
		}
	}

	int start_piece = 10;
	if (input.find("startpiece") != input.end()) {
		int p = std::stoi(input["startpiece"]);
		assert(p == 8 || p == 9 || p == 10);
		start_piece = p;
	}

	init_move_tables();

	if (input.empty()) {
		std::cout << "LOG: [" << get_datetime_str() << "] notice: no command-line input" << std::endl;
		solve_8();
		solve_8_bfs();
		std::cout << "LOG: [" << get_datetime_str() << "] finish" << std::endl;
		return 0;
	}

	if (input.find("test") != input.end()) {
		unittests();
	}

	if (input.find("retrograde") != input.end()) {
		std::cout << "LOG: [" << get_datetime_str() << "] start: retrograde analysis: start_piece = " << start_piece << std::endl;
		const auto t = std::chrono::system_clock::now();

		OstleRetrogradeAnalyzer e(start_piece);
		e.retrograde_analysis();
		e.print_statistics_of_results();
		e.answer_about_initial_position();
		e.output_positions_and_solutions("ostle_" + std::to_string(start_piece) + "_retrograde_output");

		const uint64_t fingerprint = e.calc_final_result_hashvalue();
		const auto s = std::chrono::system_clock::now();
		const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
		std::cout << "LOG: [" << get_datetime_str() << "] finish: retrograde analysis: start_piece = " << start_piece << ", elapsed time = " << elapsed << ", fingerprint = " << fingerprint << std::endl;
	}

	if (input.find("bfs") != input.end()) {
		std::cout << "LOG: [" << get_datetime_str() << "] start: breadth-first search from the initial node: start_piece = " << start_piece << std::endl;
		const auto t = std::chrono::system_clock::now();

		OstleRetrogradeAnalyzer e(start_piece);
		e.retrograde_analysis();
		e.print_statistics_of_results();
		e.answer_about_initial_position();
		e.output_positions_and_solutions("ostle_" + std::to_string(start_piece) + "_bfs_output");

		const uint64_t fingerprint = e.calc_final_result_hashvalue();
		const auto s = std::chrono::system_clock::now();
		const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
		std::cout << "LOG: [" << get_datetime_str() << "] finish: breadth-first search from the initial node: start_piece = " << start_piece << ", elapsed time = " << elapsed << ", fingerprint = " << fingerprint << std::endl;
	}

	return 0;
}

