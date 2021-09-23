#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<map>
#include<set>
#include<unordered_map>
#include<unordered_set>
#include<queue>
#include<algorithm>
#include<cassert>
#include<cstdint>
#include<regex>
#include<random>
#include<cstdint>
#include<iomanip>
#include<chrono>
#include<array>
#include<bitset>
#include<functional>
#include<limits>
#include<exception>

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
			else if(pos_hole == i * 5 + j)std::cout << "◎";
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

	t = (b ^ (b >> 7)) & 0x00aa00aa00aa00aaULL;
	b = b ^ t ^ (t << 7);
	t = (b ^ (b >> 14)) & 0x0000cccc0000ccccULL;
	b = b ^ t ^ (t << 14);
	t = (b ^ (b >> 28)) & 0x00000000f0f0f0f0ULL;
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
			std::cout << "test_checkmate_detector_func: " << i << " / " << length << std::endl;
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


class Encoder_AES {
	//自作ハッシュテーブルのハッシュ関数としてAES暗号を流用する。

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

	uint64_t dfs_position_root(const uint64_t pos_hole, const int num_piece_player, const int num_piece_opponent, std::vector<uint64_t> &all_positions, const bool PARALLEL = false) {
		//穴の位置とコマの数を引数に取り、その条件に沿った盤面を全列挙して、対称な局面を同一視して重複削除してからall_positionsの末尾に加える。返り値は、最終的に加えた盤面の数とする。

		positions.clear();
		positions.reserve(500'000'000ULL);

		dfs_position(0, 0, pos_hole, 0, num_piece_player, num_piece_opponent);

		const int64_t siz = positions.size();

		if (PARALLEL) {
#pragma omp parallel for
			for (int64_t i = 0; i < siz; ++i) {
				positions[i] = code_unique(positions[i]);
			}
		}
		else {
			for (uint64_t i = 0; i < positions.size(); ++i) {
				positions[i] = code_unique(positions[i]);
			}
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

	void do_enumerate(const int start_piece, std::vector<uint64_t> &all_positions, const bool PARALLEL = false) {

		num_start_piece = start_piece;

		const uint64_t holes[6] = { 0,1,2,6,7,12 };
		uint64_t num_unique_positions = 0;

		std::cout << "LOG: start: dfs_position_root" << std::endl;

		switch (num_start_piece) {
		case 10:
			for (int i = 0; i < 6; ++i)num_unique_positions += dfs_position_root(holes[i], 5, 5, all_positions, PARALLEL);
		case 9:
			for (int i = 0; i < 6; ++i)num_unique_positions += dfs_position_root(holes[i], 5, 4, all_positions, PARALLEL);
			for (int i = 0; i < 6; ++i)num_unique_positions += dfs_position_root(holes[i], 4, 5, all_positions, PARALLEL);
		case 8:
			for (int i = 0; i < 6; ++i)num_unique_positions += dfs_position_root(holes[i], 4, 4, all_positions, PARALLEL);
			break;
		default:
			assert(false);
		}

		std::cout << "LOG: finish: dfs_position_root" << std::endl;

		std::cout << "result: total number of the unique positions = " << num_unique_positions << std::endl;

		assert(num_unique_positions == all_positions.size());

		std::cout << "LOG: start: sort and verify uniqueness" << std::endl;

		std::sort(all_positions.begin(), all_positions.end());

		for (uint64_t i = 1; i < all_positions.size(); ++i) {
			assert(all_positions[i - 1] < all_positions[i]);
		}

		std::cout << "LOG: finish: sort and verify uniqueness" << std::endl;

		return;
	}
};

template<bool USE_HASH_TABLE, bool USE_LEVELWISE, bool PARALLEL>class OstleEnumerator {

private:

	std::vector<uint64_t>all_positions;

	std::vector<uint64_t>all_solutions;

	std::vector<uint8_t>signature_table;

	Encoder_AES hash_func;

	int num_start_piece;

	enum {
		UNLABELED = 0,
		WIN = 1,
		LOSE = 2,
		SUICIDE = 100
	};

	constexpr static uint64_t SOLUTION_ALL_WIN = 0x5555'5555'5555'5555ULL;
	constexpr static uint64_t SOLUTION_ALL_LOSE = 0xAAAA'AAAA'AAAA'AAAAULL;

	bool constract_hashtable_if_hashable(const uint64_t hash_length, const uint64_t hash_seed) {
		//ハッシュテーブルの長さとハッシュ関数のseed値を引数に取り、all_positionsがそのハッシュテーブルに収まるかどうか調べる。
		//収まらないならばfalseを返す。収まるならば、ハッシュテーブルを実際に構築してからtrueを返す。
		//返り値がfalseならば、all_positionsは変更されない。trueならば、all_positionsは変更されうる。（並べ替えられて、ゼロの要素が挿入されることがある）

		assert(USE_HASH_TABLE == true);

		//hash_lengthは2^32未満でなければならないが、2べき数でなくてもよい。
		assert(hash_length < (1ULL << 32));

		if (hash_length < all_positions.size())return false;

		//key observation:
		//エントリeを[0,2^32)のハッシュ値h(e)に一様ランダムに写像できたならば、
		//((hash_length * h(e)) >> 32) はeを[0,hash_length)に一様ランダムに写像するものである。（整数除算命令を用いずに！）
		// cf: https://github.com/official-stockfish/Stockfish/commit/2198cd0524574f0d9df8c0ec9aaf14ad8c94402b

		hash_func = Encoder_AES(hash_seed);

		//度数表を作る。この時点で、32を超える配列要素があれば、必ず構築失敗する。
		std::vector<uint8_t>count(hash_length, 0);
		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			assert(all_positions[i] < (1ULL << 55));//ここのall_positionsはunique_ostleの出力であるはず
			const uint64_t hash_key = hash_func.aes128_enc(all_positions[i]);
			const uint64_t hashtable_index = ((hash_key >> 32) * hash_length) >> 32;
			assert(hashtable_index < hash_length);
			if (++count[hashtable_index] > 32) {
				return false;
			}
		}

		//各添字番号について、（Robin Hood Hashingでinsertしたときに）その要素が実際に格納され始める位置を求める。
		//32以上離れることがあれば、それは構築失敗を意味する。
		int prev_start_pos = 0;
		for (uint64_t i = 1; i < hash_length; ++i) {
			const int current_start_pos = uint8_t(std::max(prev_start_pos + int(count[i - 1]) - 1, 0)); // main DP
			if (current_start_pos >= 32) {
				return false;
			}
			prev_start_pos = current_start_pos;
		}

		//最後の添字番号について、その要素が格納され終わる位置を求める。32以上離れることがあれば、それは構築失敗を意味する。
		if (prev_start_pos + uint64_t(count[hash_length - 1]) >= 32)return false;

		//
		//処理がここに到達したということは構築可能である。よってここからは実際の構築処理を行う。
		//

		//配列all_positionsをハッシュ値の昇順で並べ替える。
		std::sort(all_positions.begin(), all_positions.end(),
			[&](const uint64_t a, const uint64_t b) {return hash_func.aes128_enc(a) < hash_func.aes128_enc(b); });

		//配列all_positionsがハッシュテーブルの長さになるように、配列の先頭にゼロを付加する。
		assert(hash_length + 31ULL >= all_positions.size());
		const uint64_t diff_length = hash_length + 31ULL - all_positions.size();
		all_positions.resize(hash_length + 31ULL);
		for (uint64_t i = all_positions.size() - 1; i >= diff_length; --i) {
			all_positions[i] = all_positions[i - diff_length];
		}
		for (uint64_t i = 0; i < diff_length; ++i) {
			all_positions[i] = 0;
		}

		//signature_tableをハッシュテーブルの長さに初期化する。最上位ビットを立てておく（空白を意味する）。
		signature_table.clear();
		signature_table.resize(hash_length + 31ULL, 0x80U);

		//iはall_positionsの中身が入っている部分全体をなめて、各要素をできるだけ前方にずらす。
		//ただし、[table_index]より手前にならないように注意する。
		uint64_t cursor = 0;
		for (uint64_t i = diff_length; i < all_positions.size(); ++i, ++cursor) {
			const uint64_t hash_key = hash_func.aes128_enc(all_positions[i]);
			const uint64_t hashtable_index = ((hash_key >> 32) * hash_length) >> 32;
			for (; cursor < hashtable_index; ++cursor) {
				all_positions[cursor] = 0;
			}
			assert(hashtable_index <= cursor && cursor <= hashtable_index + 31ULL);
			all_positions[cursor] = all_positions[i];
			signature_table[cursor] = hash_key % 0x80ULL;
			assert(cursor <= i);
		}
		for (; cursor < all_positions.size(); ++cursor)all_positions[cursor] = 0;

		assert(all_positions.size() == signature_table.size());
		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			if (signature_table[i] == 0x80U)assert(all_positions[i] == 0);
			else {
				const uint64_t hash_key = hash_func.aes128_enc(all_positions[i]);
				assert(signature_table[i] == hash_key % 0x80ULL);
				const uint64_t c = find(all_positions[i]);
				assert(c == i);
			}
		}

		return true;
	}

	void prepare_for_hashtable() {
		//all_positionsを並べ替えつつ適切な位置に空白要素を挿入して、ハッシュテーブルとして機能するようにする。
		//挿入する空白要素の数をできるだけ少なくする（＝load factorをできるだけ大きくする）ため、トライアルアンドエラーを行う。

		assert(USE_HASH_TABLE == true);

		const uint64_t num_unique_positions = all_positions.size();

		uint64_t hash_length = num_unique_positions + (num_unique_positions >> 3);
		uint64_t hash_seed = 12345;

		for (;; hash_length += hash_length >> 3) {

			if (hash_length >= (1ULL << 32)) {
				std::cout << "LOG: failed to construct hashtable."<< std::endl;
				std::exit(0);
			}

			std::cout << "LOG: start: constract_hashtable_if_hashable: hash_length = " << hash_length << ", hash_seed =" << hash_seed << std::endl;
			const bool b = constract_hashtable_if_hashable(hash_length, hash_seed);
			if (b)break;
		}

		const double loading_factor = 100.0 * double(num_unique_positions) / double(hash_length);
		std::cout << "LOG: finish: constract_hashtable_if_hashable: hash_length = " << hash_length << ", hash_seed =" << hash_seed << ", loading factor = " << loading_factor << " %" << std::endl;
	}

	void dfs_binary_tree_and_add_level(const uint64_t index, const uint64_t level, uint64_t &counter) {

		if (index * 2 + 1 < all_positions.size()) {
			dfs_binary_tree_and_add_level(index * 2 + 1, level + 1, counter);
		}

		all_positions[counter++] |= level << 55;

		if (index * 2 + 2 < all_positions.size()) {
			dfs_binary_tree_and_add_level(index * 2 + 2, level + 1, counter);
		}
	}

	void shuffle_sorted_to_levelwise() {
		//all_positionsが昇順にソートされていると仮定して、かつ55bitで収まっていると仮定して、levelwiseに並べ替える。

		for (uint64_t i = 1; i < all_positions.size(); ++i) {
			assert(all_positions[i - 1] < all_positions[i]);
		}
		assert(all_positions.back() < (1ULL << 55));

		uint64_t counter = 0;
		dfs_binary_tree_and_add_level(0, 1, counter);
		assert(counter == all_positions.size());

		std::sort(all_positions.begin(), all_positions.end());

		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			all_positions[i] &= (1ULL << 55) - 1ULL;
		}
	}

	uint64_t find(const uint64_t code) {
		//all_positions[i]=codeなるiを探して返す。必ず見つかると仮定する。

		assert(USE_HASH_TABLE == true);

		const uint64_t hash_length = all_positions.size() - 31ULL;
		const uint64_t hash_key = hash_func.aes128_enc(code);
		const uint64_t hashtable_index = ((hash_key >> 32) * hash_length) >> 32;
		const uint8_t signature = uint8_t(hash_key % 0x80ULL);

		//ナイーブな処理手順では、all_positions[hashtable_index]から順番になめていって、所望の局面に当たったらOKとする。
		//でも今回はシグネチャ配列があるので効率的に計算できる。空白エントリはシグネチャが0x80であることを考慮しつつ、以下のようなAVX2のコードが書ける。

		__m256i query_signature = _mm256_set1_epi8(signature);
		__m256i table_signature = _mm256_loadu_si256((__m256i*)&signature_table.data()[hashtable_index]);

		//[index + i]の情報が ↑のsignature_table.i8[i]に格納されているとして、↓のi桁目のbitに移されるとする。

		const uint64_t is_empty = uint32_t(_mm256_movemask_epi8(table_signature));
		const uint64_t is_positive = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(query_signature, table_signature)));

		uint64_t to_look = (is_empty ^ (is_empty - 1)) & is_positive;

		//[index+i]がシグネチャ陽性かどうかがis_positiveの下からi番目のビットにあるとする。
		//最初に当たる空白エントリより手前にあるシグネチャ陽性な局面の位置のビットボードが計算できる。to_lookがそれである。

		for (uint32_t i = 0; bitscan_forward64(to_look, &i); to_look &= to_look - 1) {
			const uint64_t pos = hashtable_index + i;
			if (all_positions[pos] == code)return pos;
		}

		assert(false);
		return 0;
	}

	uint64_t code2index(const uint64_t code) {
		//all_positionsのどれかの値codeを引数にとり、all_positions[i]=codeなるiを探して返す。
		//all_positionsが昇順にソートされていると仮定して二分探索で求める。

		assert(USE_HASH_TABLE == false);
		assert(USE_LEVELWISE == false);

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

	uint64_t code2index_levelwise(const uint64_t code) {
		//all_positionsのどれかの値cを引数にとり、all_positions[i]=cなるiを探して返す。
		//all_positionsが昇順にソートされてからlevelwiseに並べ替えられていると仮定して二分探索で求める。

		assert(USE_HASH_TABLE == false);
		assert(USE_LEVELWISE == true);

		uint64_t i = 0;
		while (all_positions[i] != code) {
			if (all_positions[i] < code) {
				i = i * 2 + 2;
			}
			else {
				i = i * 2 + 1;
			}
			assert(i < all_positions.size());
		}
		return i;
	}


	void check_all_positions_if_checkmate() {
		//all_positionsの全ての盤面について、手番側が即座に勝利できる局面かどうか調べて、そうならばall_solutionsを更新する。

		if (USE_HASH_TABLE) {
			//ハッシュテーブルありの場合、ハッシュテーブルが既に構築済みであることを仮定する。（＝constract_hashtable_if_hashableを呼んでtrueが返ってきている）
			assert(all_positions.size() == signature_table.size());
		}

		assert(all_positions.size() == all_solutions.size());

		std::cout << "LOG: start: check_all_positions_if_checkmate" << std::endl;

		uint64_t count_checkmate_position = 0, count_position = 0;
		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			if (USE_HASH_TABLE) {
				if (signature_table[i] & 0x80U)continue;
			}
			++count_position;
			uint64_t bb1 = 0, bb2 = 0, pos = 0;
			decode_ostle(all_positions[i], bb1, bb2, pos);
			if (is_checkmate(bb1, bb2, pos)) {
				all_solutions[i] = SOLUTION_ALL_WIN;
				++count_checkmate_position;
			}
		}

		const double percentage = 100.0 * double(count_checkmate_position) / double(count_position);

		std::cout << "result: number of checkmate positions = " << count_checkmate_position << " / " << count_position
			<< " (" << percentage << " %)" << std::endl;
	}

	uint64_t check_one_position(const uint64_t index) {
		//盤面all_positions[index]の全てのノード（禁じ手によって区別される）について勝敗判定を試みる。
		//返り値は、今回の試行で新たに勝敗判定がなされたなら、更新した判定結果を返す。さもなくば更新されなかった判定結果を返す。

		if (USE_HASH_TABLE) {
			if (signature_table[index] & 0x80U)return 0ULL;
		}

		if (all_solutions[index] == SOLUTION_ALL_WIN ||
			all_solutions[index] == SOLUTION_ALL_LOSE) {
			return all_solutions[index];
		}

		uint64_t next_all_solutions = all_solutions[index];

		//処理がここに到達した時点で、盤面all_positions[index]はcheckmate盤面ではないと仮定する。
		//言い換えると、予めcheck_all_positions_if_checkmate関数が呼ばれていることを仮定する。

		uint64_t bb_player = 0, bb_opponent = 0, pos_hole = 0;
		decode_ostle(all_positions[index], bb_player, bb_opponent, pos_hole);

		Moves moves, next_moves;
		generate_moves(bb_player, bb_opponent, pos_hole, moves);

		assert(moves[0] <= 24);

		//まず、各指し手を指した先の局面が勝敗判定されているか調べる。
		uint8_t outcome[32] = {}, win_num = 0, lose_num = 0, unknown_num = 0, suicide_num = 0;
		for (uint8_t i = 1; i <= moves[0]; ++i) {
			uint64_t next_bb_player = bb_player, next_bb_opponent = bb_opponent, next_pos_hole = pos_hole;
			do_move(next_bb_player, next_bb_opponent, next_pos_hole, moves[i]);
			if (_mm_popcnt_u64(next_bb_player) == 3) {
				outcome[i] = SUICIDE;
				++suicide_num;
				continue;
			}
			const uint64_t next_code = code_unique(encode_ostle(next_bb_opponent, next_bb_player, next_pos_hole));
			const uint64_t next_index =
				USE_HASH_TABLE ?
				find(next_code) :
				(USE_LEVELWISE ? code2index_levelwise(next_code) : code2index(next_code));

			if (all_solutions[next_index] == SOLUTION_ALL_WIN) {
				outcome[i] = LOSE;
				++lose_num;
				continue;
			}
			if (all_solutions[next_index] == SOLUTION_ALL_LOSE) {
				outcome[i] = WIN;
				++win_num;
				continue;
			}
			if (all_solutions[next_index] == 0) {
				outcome[i] = UNLABELED;
				++unknown_num;
				continue;
			}

			generate_moves(next_bb_opponent, next_bb_player, next_pos_hole, next_moves);
			int forbidden_index = 0;
			for (uint8_t j = 1; j <= next_moves[0]; ++j) {
				uint64_t bb1 = next_bb_opponent, bb2 = next_bb_player, pos = next_pos_hole;
				do_move(bb1, bb2, pos, next_moves[j]);
				if (_mm_popcnt_u64(bb1) == 3)continue;
				if (bb1 == bb_player && bb2 == bb_opponent && pos == pos_hole) {
					forbidden_index = j;
					break;
				}
			}
			const uint64_t next_solution = (all_solutions[next_index] >> (forbidden_index * 2)) % 4;
			if (next_solution == LOSE) {
				outcome[i] = WIN;
				if (++win_num >= 2)break;
			}
			else if (next_solution == WIN) {
				outcome[i] = LOSE;
				++lose_num;
			}
			else {
				assert(next_solution == UNLABELED);
				outcome[i] = UNLABELED;
				++unknown_num;
			}

		}

		//2つ以上の指し手が勝利をもたらすなら、禁じ手がどれであろうと勝利が確定する。
		if (win_num >= 2) {
			return SOLUTION_ALL_WIN;
		}

		assert(int(win_num) + int(lose_num) + int(unknown_num) + int(suicide_num) == int(moves[0]));

		//1つの指し手Xだけが勝利確定手の場合
		if (win_num == 1) {
			for (uint8_t i = 0; i <= moves[0]; ++i) {
				const uint8_t present_solution = uint8_t((all_solutions[index] >> (i * 2)) % 4);
				if (i == 0 || outcome[i] != WIN) {
					//禁じ手が存在しないか、禁じ手がX以外である場合は、（Xを指せばいいので）勝利が確定する。
					assert(present_solution == UNLABELED || present_solution == WIN);
					if (present_solution == UNLABELED) {
						next_all_solutions |= uint64_t(WIN) << (i * 2);
					}
				}
				else {
					//Xが禁じ手である場合、X以外の指し手が全て敗北確定ならば「Xが禁じ手である場合は敗北」と確定する。さもなくば未確定である。
					if (unknown_num == 0) {
						assert(present_solution == UNLABELED || present_solution == LOSE);
						if (present_solution == UNLABELED) {
							next_all_solutions |= uint64_t(LOSE) << (i * 2);
						}
					}
					else {
						assert(present_solution == UNLABELED);
					}
				}
			}
			return next_all_solutions;
		}

		//全ての指し手が敗北確定なら、禁じ手がどれであろうと敗北が確定する。
		if (win_num == 0 && unknown_num == 0) {
			return SOLUTION_ALL_LOSE;
		}

		//ただ1つの指し手Xが未確定で、それ以外全ての指し手が敗北確定の場合
		if (win_num == 0 && unknown_num == 1) {
			for (uint8_t i = 0; i <= moves[0]; ++i) {
				const uint64_t present_solution = (all_solutions[index] >> (i * 2)) % 4;
				if (i == 0) {
					//禁じ手が存在しない場合は（Xを指せるので）未確定。
					assert(present_solution == UNLABELED);
					continue;
				}
				else if (outcome[i] == UNLABELED) {
					//Xが禁じ手である場合は敗北が確定する。
					assert(present_solution == UNLABELED || present_solution == LOSE);
					if (present_solution == UNLABELED) {
						next_all_solutions |= uint64_t(LOSE) << (i * 2);
					}
				}
				else {
					//X以外が禁じ手である場合は（Xを指せるので）未確定。

					assert(outcome[i] == LOSE || outcome[i] == SUICIDE);
					assert(present_solution == UNLABELED);
				}
			}
			return next_all_solutions;
		}

		//ここに到達したということは、勝利確定手が見つかっておらず、かつ未確定の指し手が2つ以上存在する。
		assert(win_num == 0 && unknown_num >= 2);

		//その場合は禁じ手がどれであろうと未確定である。
		assert(all_solutions[index] == 0);
		return 0;
	}

	uint64_t retrograde_analysis_single_iteration_serial() {
		uint64_t updated_num = 0;
		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			const uint64_t next_solution = check_one_position(i);
			updated_num += (next_solution != all_solutions[i]) ? 1 : 0;
			all_solutions[i] = next_solution;
		}
		return updated_num;
	}

	uint64_t retrograde_analysis_single_iteration_parallel() {

		constexpr uint64_t TASK_SIZE = 1'000'000ULL;

		std::vector<uint64_t>next_solutions;
		next_solutions.reserve(TASK_SIZE);

		uint64_t updated_num = 0;

		for (uint64_t t = 0; t < all_positions.size(); t += TASK_SIZE) {

			next_solutions.resize(std::min(TASK_SIZE, all_positions.size() - t));

			const int64_t start = t, end = t + next_solutions.size();

#pragma omp parallel for schedule(guided)
			for (int64_t i = start; i < end; ++i) {
				const uint64_t next_solution = check_one_position(i);
				next_solutions[i - start] = next_solution;
			}

			for (int64_t i = start; i < end; ++i) {
				updated_num += (next_solutions[i - start] != all_solutions[i]) ? 1 : 0;
				all_solutions[i] = next_solutions[i - start];
			}

		}
		return updated_num;
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

	OstleEnumerator() :OstleEnumerator(8) {}

	void do_enumerate() {
		PositionEnumerator p;
		p.do_enumerate(num_start_piece, all_positions, PARALLEL);
	}

	void retrograde_analysis() {

		std::cout << "LOG: start: retrograde_analysis" << std::endl;


		if (USE_HASH_TABLE) {
			prepare_for_hashtable();
		}
		else {
			if (USE_LEVELWISE) {
				shuffle_sorted_to_levelwise();
			}
		}

		all_solutions.clear();
		all_solutions.resize(all_positions.size());

		check_all_positions_if_checkmate();

		for(int iteration = 1;; ++iteration) {
			std::cout << "LOG: start: iteration " << iteration << std::endl;
			const auto t = std::chrono::system_clock::now();

			const uint64_t updated_num =
				PARALLEL ?
				retrograde_analysis_single_iteration_parallel() :
				retrograde_analysis_single_iteration_serial();

			const auto s = std::chrono::system_clock::now();
			const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
			std::cout << "LOG: finish: iteration " << iteration << " : updated_num = " << updated_num << ", elapsed time = " << elapsed << " milliseconds" << std::endl;
			if (updated_num == 0)break;
		}

		std::cout << "LOG: finish: retrograde_analysis" << std::endl;
	}

	uint64_t calc_final_result_hashvalue() {
		//全盤面とその解析結果の組に関するハッシュ値を生成して返す。
		//USE_HASH_TABLE, USE_LEVELWISE, PARALLELの有無によって計算結果が変わらないはずだが、これを確かめるのに使う。
		//USE_HASH_TABLE, USE_LEVELWISEの有無によってall_positionsの中身の順番が異なるので、可換な演算でreduceする必要がある。

		assert(all_positions.size() == all_solutions.size());
		if (USE_HASH_TABLE) {
			assert(all_positions.size() == signature_table.size());
		}

		Encoder_AES encoder(123456);

		__m128i answer = _mm_setzero_si128();

		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			if (USE_HASH_TABLE) {
				if (signature_table[i] & 0x80U)continue;
			}
			const __m128i x = encoder.aes128_enc(all_positions[i], all_solutions[i]);
			answer = _mm_xor_si128(x, answer);//xorは可換
		}

		uint64_t a[2];
		_mm_storeu_si128((__m128i*)a, answer);
		return a[0] ^ a[1];
	}

	void output_results(const std::string filename) {

		assert(all_positions.size() == all_solutions.size());
		if (USE_HASH_TABLE) {
			assert(all_positions.size() == signature_table.size());
		}

		std::ofstream writing_file;
		uint64_t count = 0;
		std::string code1, code2;
		constexpr uint64_t SINGLE_FILE_LIMIT = 1000000;

		for (uint64_t i = 0; i < all_positions.size(); ++i) {
			if (USE_HASH_TABLE) {
				if (signature_table[i] & 0x80U)continue;
			}
			if (count % SINGLE_FILE_LIMIT == 0) {
				if (count) {
					writing_file.close();
				}
				writing_file.open(filename + my_itos(count / SINGLE_FILE_LIMIT, 4, '0') + std::string(".txt"), std::ios::out);
			}
			encode_base64(all_positions[i], code1);
			encode_base64(all_solutions[i], code2);
			writing_file << code1 << "," << code2 << std::endl;
			++count;
		}
		if (all_positions.size()) {
			writing_file.close();
		}
	}
};

uint64_t enumerate_binarysearch_parallel() {
	std::cout << "LOG: start: enumerate_binarysearch_parallel" << std::endl;
	OstleEnumerator<false, true, true> e;
	e.do_enumerate();
	e.retrograde_analysis();
	const uint64_t fingerprint = e.calc_final_result_hashvalue();
	std::cout << "LOG: finish: enumerate_binarysearch_parallel. fingerprint = " << fingerprint << std::endl;
	return fingerprint;
}

uint64_t enumerate_binarysearch_serial() {
	std::cout << "LOG: start: enumerate_binarysearch_serial" << std::endl;
	OstleEnumerator<false, true, false> e;
	e.do_enumerate();
	e.retrograde_analysis();
	const uint64_t fingerprint = e.calc_final_result_hashvalue();
	std::cout << "LOG: finish: enumerate_binarysearch_serial. fingerprint = " << fingerprint << std::endl;
	return fingerprint;
}

uint64_t enumerate_hashtable_parallel() {
	std::cout << "LOG: start: enumerate_hashtable_parallel" << std::endl;
	OstleEnumerator<true, true, true> e;
	e.do_enumerate();
	e.retrograde_analysis();
	const uint64_t fingerprint = e.calc_final_result_hashvalue();
	std::cout << "LOG: finish: enumerate_hashtable_parallel. fingerprint = " << fingerprint << std::endl;
	return fingerprint;
}

uint64_t enumerate_hashtable_serial() {
	std::cout << "LOG: start: enumerate_hashtable_serial" << std::endl;
	OstleEnumerator<true, true, false> e;
	e.do_enumerate();
	e.retrograde_analysis();
	const uint64_t fingerprint = e.calc_final_result_hashvalue();
	std::cout << "LOG: finish: enumerate_hashtable_serial. fingerprint = " << fingerprint << std::endl;
	return fingerprint;
}

void test_all_strategies() {
	const uint64_t fingerprint1 = enumerate_binarysearch_parallel();
	const uint64_t fingerprint2 = enumerate_binarysearch_serial();
	const uint64_t fingerprint3 = enumerate_hashtable_parallel();
	const uint64_t fingerprint4 = enumerate_hashtable_serial();
	assert(fingerprint1 == fingerprint2);
	assert(fingerprint1 == fingerprint3);
	assert(fingerprint1 == fingerprint4);
	std::cout << "test clear!" << std::endl;
}







uint64_t shiftxor_forward(uint64_t x, int s) {
	return x ^ (x >> s);
}

uint64_t split_mix_64(uint64_t x) {

	// SplitMix64
	// http://prng.di.unimi.it/splitmix64.c

	x += 0x9e3779b97f4a7c15ULL;
	x = shiftxor_forward(x, 30);
	x *= 0xbf58476d1ce4e5b9ULL;
	x = shiftxor_forward(x, 27);
	x *= 0x94d049bb133111ebULL;
	x = shiftxor_forward(x, 31);
	return x;
}


void dfs_binary_tree(const uint64_t index, const uint64_t level, uint64_t &counter, std::vector<uint64_t> &v) {

	if (index * 2 + 1 < v.size()) {
		dfs_binary_tree(index * 2 + 1, level + 1, counter, v);
	}

	v[counter++] |= level << 55;

	if (index * 2 + 2 < v.size()) {
		dfs_binary_tree(index * 2 + 2, level + 1, counter, v);
	}
}

void shuffle_sorted_to_levelwise(std::vector<uint64_t> &v) {
	//vが昇順にソートされていると仮定して、かつ55bitで収まっていると仮定して、levelwiseに並べ替える。

	for (uint64_t i = 1; i < v.size(); ++i) {
		assert(v[i - 1] < v[i]);
	}
	assert(v.back() < (1ULL << 55));

	uint64_t counter = 0;
	dfs_binary_tree(0, 1, counter, v);
	assert(counter == v.size());

	std::sort(v.begin(), v.end());

	for (uint64_t i = 0; i < v.size(); ++i) {
		v[i] &= (1ULL << 55) - 1ULL;
	}
}


uint64_t c2i(const uint64_t c, const std::vector<uint64_t> &table) {
	//tableのどれかの値cを引数にとり、table[i]=cなるiを探して返す。
	//tableが昇順にソートされていると仮定して二分探索で求める。

	uint64_t lower = 0, upper = table.size();

	while (lower + 1ULL < upper) {
		const uint64_t mid = (lower + upper) / 2;
		if (table[mid] == c)return mid;
		else if (table[mid] <= c)lower = mid;
		else upper = mid;
	}

	assert(table[lower] == c);
	return lower;
}

uint64_t c2i_levelwise (const uint64_t c, const std::vector<uint64_t> &table) {
	//tableのどれかの値cを引数にとり、table[i]=cなるiを探して返す。
	//tableが昇順にソートされてからlevelwiseに並べ替えられていると仮定して二分探索で求める。

	uint64_t i = 0;
	while(table[i] != c) {
		if (table[i] < c) {
			i = i * 2 + 2;
		}
		else {
			i = i * 2 + 1;
		}
		assert(i < table.size());
	}
	return i;
}

template<bool LEVELWISE> void speedtest_c2i(const uint64_t size, const uint64_t num_trial, const uint64_t seed) {

	std::vector<uint64_t>test_table(size);

	test_table[0] = seed;
	for (uint64_t i = 1; i < size; ++i) {
		test_table[i] = split_mix_64(test_table[i - 1]);
	}
	for (uint64_t i = 0; i < size; ++i) {
		test_table[i] &= (1ULL << 55) - 1ULL;
	}

	std::sort(test_table.begin(), test_table.end());

	for (uint64_t i = 1; i < size; ++i) {
		assert(test_table[i - 1] < test_table[i]);
	}

	if (LEVELWISE) {
		shuffle_sorted_to_levelwise(test_table);
	}

	std::cout << "LOG: start: speedtest_c2i : " << (LEVELWISE ? "LEVELWISE" : "NOT LEVELWISE") <<std::endl;
	const auto t = std::chrono::system_clock::now();

	const uint64_t N = std::min(num_trial, size);
	uint64_t result = 0;
	for (uint64_t i = 0, c = seed; i < N; ++i, c = split_mix_64(c)) {
		const uint64_t x = LEVELWISE ? c2i_levelwise(c & ((1ULL << 55) - 1ULL), test_table) : c2i(c & ((1ULL << 55) - 1ULL), test_table);
		result += x;
	}

	const auto s = std::chrono::system_clock::now();
	const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
	std::cout << "LOG: result = " << result << ", size = " << size << ", num_trial = " << N << ", elapsed time = " << elapsed << " milliseconds" << std::endl;
}

void speedtest_binarysearch() {
	const uint64_t size = 100'000'000ULL;
	const uint64_t num_trial = 100'000'000ULL;
	speedtest_c2i<true>(size, num_trial, 12345);
	speedtest_c2i<false>(size, num_trial, 12345);
}



void unittests() {

	test_move(12345, 100000);
	test_checkmate_detector_func(12345, 100000);
	test_bitboard_symmetry(12345, 100000);
	test_base64_func(12345, 100000);

	std::cout << "test clear!" << std::endl;
	std::exit(0);
}



int main(int argc, char *argv[]) {

	//uint64_tがなにかのエイリアスであることは規定されているが、unsigned long longのエイリアスであるとは限らない。
	//処理系によってはlongが64bitでuint64_tがunsigned longのエイリアスであることもある。（言語仕様で許容されている）
	//整数リテラルの末尾にULLを付けるとunsigned long long型であることが明示される。uint64_tではなくunsigned long longになることが問題である。
	//例えばstd::max関数は2つの引数が同じ型でなければならないのだが、uint64_tとunsigned long longを入れたときにコンパイルエラーになる可能性が処理系によってありうる。
	//以下のstatic_assertは、そういう処理系を検知してコンパイルエラーにする。
	static_assert(std::is_same<uint64_t, unsigned long long>::value);
	static_assert(std::is_same<int64_t, long long>::value);
	static_assert(std::is_same<uint32_t, unsigned int>::value);
	static_assert(std::is_same<int32_t, int>::value);
	static_assert(std::is_same<uint64_t, size_t>::value);

	init_move_tables();

	if (argc == 3 && argv[1] == "go" && (argv[2] == "parallel" || argv[2] == "serial")) {
		std::cout << "LOG: start: analyze the all positions" << std::endl;
		const auto t = std::chrono::system_clock::now();

		uint64_t fingerprint = 0;
		if (argv[2] == "parallel") {
			OstleEnumerator<false, false, true>e(10);
			e.do_enumerate();
			e.retrograde_analysis();
			e.output_results("ostle_output");
			fingerprint = e.calc_final_result_hashvalue();
		}
		else {
			OstleEnumerator<false, false, false>e(10);
			e.do_enumerate();
			e.retrograde_analysis();
			e.output_results("ostle_output");
			fingerprint = e.calc_final_result_hashvalue();
		}
		const auto s = std::chrono::system_clock::now();
		const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
		std::cout << "LOG: finish: analyze the all positions: elapsed time = " << elapsed << ", fingerprint = " <<fingerprint << std::endl;

		return 0;
	}

	//speedtest_binarysearch();

	//unittests();

	//test_all_strategies();

	{
		const auto t = std::chrono::system_clock::now();
		OstleEnumerator<false, true, true>e(8);
		e.do_enumerate();
		e.retrograde_analysis();
		//e.output_results("ostle_output");
		const auto s = std::chrono::system_clock::now();
		const auto n = e.calc_final_result_hashvalue();
		const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
		std::cout << "LEVELWISE: time = " << elapsed << ", fingerprint = " << n << std::endl;
	}
	{
		const auto t = std::chrono::system_clock::now();
		OstleEnumerator<false, false, true>e(8);
		e.do_enumerate();
		e.retrograde_analysis();
		//e.output_results("ostle_output");
		const auto s = std::chrono::system_clock::now();
		const auto n = e.calc_final_result_hashvalue();
		const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(s - t).count();
		std::cout << "NOT LEVELEISE: time = " << elapsed << ", fingerprint = " << n << std::endl;
	}


	return 0;
}

