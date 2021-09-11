#include<iostream>
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
#include<execution>

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
	assert(pos_hole < 25);
	assert((bb_player & BB_ALL_8X8_5X5) == bb_player);
	assert((bb_opponent & BB_ALL_8X8_5X5) == bb_opponent);

	uint64_t answer = pos_hole;
	answer |= pext_intrinsics(bb_player, BB_ALL_8X8_5X5) << 5;
	answer |= pext_intrinsics(bb_opponent, BB_ALL_8X8_5X5) << 30;

	return answer;
}
void decode_ostle(const uint64_t code, uint64_t &bb_player, uint64_t &bb_opponent, uint64_t &pos_hole) {
	assert(code < (1ULL << 55));
	assert((code % 32) < 25);

	pos_hole = code % 32;
	bb_player = pdep_intrinsics(code >> 5, BB_ALL_8X8_5X5);
	bb_opponent = pdep_intrinsics(code >> 30, BB_ALL_8X8_5X5);
}

std::string code2string(const uint64_t code) {

	const uint64_t pos_hole = code % 32;

	std::string answer = "";

	for (int i = 0; i < 25; ++i) {
		if (code & (1ULL << (i + 5))) {
			answer += "1";
		}
		else if (code & (1ULL << (i + 30))) {
			answer += "2";
		}
		else if (pos_hole == i) {
			answer += "3";
		}
		else {
			answer += "0";
		}
	}
	return answer;
}

uint64_t code_symmetry_naive(const int s, uint64_t code) {

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

uint64_t code_symmetry(const int s, uint64_t code) {

	if (s & 1)code = horizontal_mirror_ostle_5x5_bitboard(code);
	if (s & 2)code = vertical_mirror_ostle_5x5_bitboard(code);
	if (s & 4)code = transpose_ostle_5x5_bitboard(code);

	return code;
}

uint64_t code_unique_naive(const uint64_t code) {

	uint64_t answer = code;

	for (int i = 1; i <= 7; ++i) {
		const uint64_t new_code = code_symmetry_naive(i, code);
		answer = std::min(answer, new_code);
	}

	return answer;
}

uint64_t code_unique(const uint64_t code) {

	uint64_t answer = code;

	for (int i = 1; i <= 7; ++i) {
		const uint64_t new_code = code_symmetry(i, code);
		answer = std::min(answer, new_code);
	}

	return answer;
}

std::string code_2_unique_string(const uint64_t code) {

	std::string answer = code2string(code);

	for (int i = 1; i <= 7; ++i) {
		const std::string new_answer = code2string(code_symmetry(i, code));
		answer = std::min(answer, new_answer);
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

constexpr uint64_t pos_diff[4] = { -5, 5, -1, 1 };

alignas(32) const static uint8_t pos_2_8x8_5x5_table[32] = { 0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };


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

uint8_t do_move_table[2][5][6][32][32][3] = {};//[a][b][c][d][e][f] a:プラス方向かマイナス方向か b:着手位置 c:穴の有無と位置 d:playerのbitboard e:opponentのbitboard f:着手後のplayer,opponentのbbと補助情報
uint8_t undo_move_table[2][5][6][32][32][6][2] = {};//[a][b][c][d][e][f][g] a:プラス方向かマイナス方向か b:着手位置 c:穴の有無と位置 d:playerのbitboard e:opponentのbitboard f:補助情報 g:着手前のplayer,opponentのbb
//2*5*6*32*32*(3+6*2)=921,600.  1メガバイト程度のテーブルになる。

void init_move_tables() {

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

					undo_move_table[a][b][c][bb_after_player][bb_after_opponent][auxiliary][0] = d;
					undo_move_table[a][b][c][bb_after_player][bb_after_opponent][auxiliary][1] = e;
				}
			}
		}
	}
}

void generate_moves(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole, Moves& moves) {
	//合法手を全列挙する。
	//直前の局面に戻る手は反則だが、ここでは気にせず生成する。

	assert((bb_player & BB_ALL_8X8_5X5) == bb_player);
	assert((bb_opponent & BB_ALL_8X8_5X5) == bb_opponent);
	assert((bb_player & bb_opponent) == 0);
	assert(pos_hole < 25);

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

	//穴を動かす手の場合。合法手であることは生成関数が保証しているので、ただ動かして終了。
	if (pos_hole == (move % 32)) {
		pos_hole += pos_diff[move / 32];
		return 0xFF;
	}

	const uint64_t bb_hole = pdep_intrinsics(1ULL << pos_hole, BB_ALL_8X8_5X5);

	if (move / 64) {

		//横方向に動かす手の場合

		const auto index1_dir = (move / 32) % 2;
		const auto index2_pos = ((move % 32) % 5);
		const auto y_pos = (move % 32) / 5;
		const auto offset_bb = y_pos * 8;
		const auto index3_hole = oneline_bb2index[(bb_hole >> offset_bb) % 256];
		const auto index4_player = (bb_player >> offset_bb) % 256;
		const auto index5_opponent = (bb_opponent >> offset_bb) % 256;

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

		const auto index1_dir = (move / 32) % 2;
		const auto index2_pos = ((move % 32) / 5);
		const auto x_pos = (move % 32) % 5;
		const auto index3_hole = oneline_bb2index[pext_intrinsics(bb_hole, BB_ONELINE_VERTICAL_8X8_5X5[x_pos])];
		const auto index4_player = pext_intrinsics(bb_player, BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);
		const auto index5_opponent = pext_intrinsics(bb_opponent, BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);

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

	//穴を動かす手の場合。合法手であることは生成関数が保証しているので、ただ動かして終了。
	if (auxiliary == 0xFF) {
		pos_hole -= pos_diff[move / 32];
		return;
	}

	const uint64_t bb_hole = pdep_intrinsics(1ULL << pos_hole, BB_ALL_8X8_5X5);

	if (move / 64) {

		//横方向に動かす手の場合

		const auto index1_dir = (move / 32) % 2;
		const auto index2_pos = ((move % 32) % 5);
		const auto y_pos = (move % 32) / 5;
		const auto offset_bb = y_pos * 8;
		const auto index3_hole = oneline_bb2index[(bb_hole >> offset_bb) % 256];
		const auto index4_player = (bb_player >> offset_bb) % 256;
		const auto index5_opponent = (bb_opponent >> offset_bb) % 256;

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

		const auto index1_dir = (move / 32) % 2;
		const auto index2_pos = ((move % 32) / 5);
		const auto x_pos = (move % 32) % 5;
		const auto index3_hole = oneline_bb2index[pext_intrinsics(bb_hole, BB_ONELINE_VERTICAL_8X8_5X5[x_pos])];
		const auto index4_player = pext_intrinsics(bb_player, BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);
		const auto index5_opponent = pext_intrinsics(bb_opponent, BB_ONELINE_VERTICAL_8X8_5X5[x_pos]);

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

bool is_checkmate_1(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole) {
	//player(手番側)にある指し手cが存在して、cを指すと相手のコマが3個になる⇔即勝利局面である⇔trueを返す。

	if (_mm_popcnt_u64(bb_opponent) != 4)return false;

	const uint64_t bb_hole = pdep_intrinsics(1ULL << pos_hole, BB_ALL_8X8_5X5);

	//key observation:
	//相手のコマを横プラス方向に押し出すことができる⇔自分のコマのbitboardを1bit左シフトして、相手のコマのbitboardに加算すれば、繰り上がりにより場外のビットが立つ。
	//(ただし左シフトした瞬間に場外に出るコマは加算の前に除外しておく必要がある)
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
		const bool b2 = is_checkmate_1(bb1, bb2, pos);
		const bool b3 = is_checkmate(bb1, bb2, pos);

		if (b1 != b2 || b1 != b3) {
			std::cout << "test failed." << std::endl;
			visualize_ostle(bb1, bb2, pos);
			const bool b1_ = is_checkmate_naive(bb1, bb2, pos, true);
			const bool b2_ = is_checkmate_1(bb1, bb2, pos);
			const bool b3_ = is_checkmate(bb1, bb2, pos);
			return false;
		}

	}
	std::cout << "test clear!" << std::endl;
	return true;
}





class HashTable {

private:

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

	uint64_t shiftxor_backward(uint64_t x, int s) {
		if (s >= 32)return x ^ (x >> s);
		for (uint64_t mask = ~(0xFFFF'FFFF'FFFF'FFFFULL >> s); mask; mask >>= s) {
			uint64_t y = x & mask;
			uint64_t z = y >> s;
			x = x ^ z;
		}
		return x;
	}

	uint64_t split_mix_64_inverse(uint64_t x) {
		x = shiftxor_backward(x, 31);
		x *= 0x319642b2d24d8ec3ULL;
		x = shiftxor_backward(x, 27);
		x *= 0x96de1b173f119089ULL;
		x = shiftxor_backward(x, 30);
		x -= 0x9e3779b97f4a7c15ULL;
		return x;
	}

	std::vector<uint64_t>hash_table;
	std::vector<uint8_t>signature_table;
	int64_t table_bitlen; // hash_tableの現在の容量は2^(table_bitlen)
	int64_t population;

	void extend() {

		std::cout << "extend: hash_table.size() = " << hash_table.size() << ", population = " << population << std::endl;

		std::vector<uint64_t>old_data(population);
		int64_t count = 0;
		for (uint64_t i = 0; i < signature_table.size(); ++i) {
			if (signature_table[i] == 0x80)continue;
			old_data[count++] = hash_table[i];
		}

		assert(count == population);

		for (; table_bitlen < 63;) {

			++table_bitlen;

			const uint64_t bitmask = (1ULL << table_bitlen) - 1;

			bool is_hopeless = false;

			//度数表を作る。この時点で、32を超える配列要素があれば、insert失敗するのでcontinueする。
			std::vector<uint8_t>count((1ULL << table_bitlen), 0);
			for (int64_t i = 0; i < population; ++i) {
				if (++count[old_data[i] & bitmask] > 32) {
					is_hopeless = true;
					break;
				}
			}
			if (is_hopeless)continue;

			//各添字番号について、（Robin Hood Hashingでinsertしたときに）その要素が実際に格納され始める位置を求める。
			//32以上離れることがあれば、それはinsert失敗を意味するのでcontinueする。

			std::vector<uint8_t>start_pos((1ULL << table_bitlen), 0);
			for (uint64_t i = 1; i <= bitmask; ++i) {
				start_pos[i] = uint8_t(std::max(uint64_t(start_pos[i - 1]) + (i - 1) + uint64_t(count[i - 1]), i) - i); // main DP
				if (start_pos[i] >= 32) {
					is_hopeless = true;
					break;
				}
			}
			if (is_hopeless)continue;
			if (uint64_t(start_pos[bitmask]) + bitmask + uint64_t(count[bitmask]) >= (1ULL << table_bitlen) + 32)continue;

			//
			//insertが失敗しないことが分かったので、ハッシュテーブルを構築する。
			//

			const uint64_t N = (1ULL << table_bitlen) + 31;
			hash_table.resize(N);
			signature_table.resize(N);
			for (int i = 0; i < N; ++i) {
				hash_table[i] = 0;
				signature_table[i] = 0x80;
			}

			for (int i = 0; i < population; ++i) {
				const uint64_t hashcode = old_data[i] & bitmask;
				hash_table[uint64_t(start_pos[hashcode]) + hashcode] = old_data[i];
				signature_table[uint64_t(start_pos[hashcode]) + hashcode] = uint8_t(old_data[i] >> 57);
				assert(start_pos[hashcode] < 32ULL);
				++start_pos[hashcode];
			}

			return;
		}

		std::cout << "error: failed to extend a hashtable" << std::endl;
		std::exit(EXIT_FAILURE);
	}

public:

	HashTable(const uint64_t bitlen) {
		table_bitlen = bitlen;

		hash_table = std::vector<uint64_t>(((1ULL << table_bitlen) + 31), 0);
		signature_table = std::vector<uint8_t>((1ULL << table_bitlen) + 31, 0x80);
		population = 0;
	}

	HashTable() :HashTable(8) {}

	void clear() {
		table_bitlen = 8;

		hash_table = std::vector<uint64_t>(((1ULL << table_bitlen) + 31), 0);
		signature_table = std::vector<uint8_t>((1ULL << table_bitlen) + 31, 0x80);
		population = 0;
	}

	int64_t size() {
		return population;
	}

	void insert(const uint64_t entry) {

		//引数の情報をハッシュテーブルに格納する、またはアップデートする。
		//衝突処理はオープンアドレス法として、indexを1ずつ増やす。Robin Hood Hashingは採用しない。本来のindexが埋まっていたら最大31まで増やすことを許容する。
		//格納もアップデートもできなかったらハッシュテーブルの大きさを倍にしてinsertしなおす。

		const uint64_t hashcode = split_mix_64(entry);
		const uint64_t index = hashcode & ((1ULL << table_bitlen) - 1);
		const uint8_t signature = uint8_t(hashcode >> 57);

		__m256i query_signature = _mm256_set1_epi8(signature);
		__m256i table_signature = _mm256_loadu_si256((__m256i*)&signature_table.data()[index]);

		//[index + i]の情報が ↑のsignature_table.i8[i]に格納されているとして、↓のi桁目のbitに移されるとする。

		const uint64_t is_empty = uint32_t(_mm256_movemask_epi8(table_signature));
		const uint64_t is_positive = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(query_signature, table_signature)));

		uint64_t to_look = (is_empty ^ (is_empty - 1)) & is_positive;

		//[index+i]がシグネチャ陽性かどうかがis_positiveの下からi番目のビットにあるとする。
		//最初に当たる空白エントリより手前にあるシグネチャ陽性な局面の位置のビットボードが計算できる。to_lookがそれである。

		//引数のエントリが既に格納されていれば、何もせずreturnする。
		for (uint32_t i = 0; bitscan_forward64(to_look, &i); to_look &= to_look - 1) {
			const uint64_t pos = index + i;
			if (hash_table[pos] == hashcode)return;
		}

		//空白エントリがあれば、（その手前に同じエントリはなかったので）、最初に見つけた空白エントリに代入して終了。
		uint32_t i = 0;
		if (bitscan_forward64(is_empty, &i)) {

			const uint64_t pos = index + i;
			signature_table[pos] = signature;
			hash_table[pos] = hashcode;
			++population;
			return;
		}
		
		//ハッシュテーブルが詰まっていて代入できなかったので、リハッシュして再試行する。
		extend();
		insert(entry);
	}

	bool find(const uint64_t entry) {

		//ハッシュテーブルに引数局面の情報があるか調べて、あればtrueを返し、なければfalseを返す。

		//ナイーブな処理手順では、[index]から順番になめていって所望の局面かどうかを調べていき、所望の局面を得るか空白エントリに当たったら終了する。
		//でも今回はシグネチャ配列があるので効率的に計算できる。空白エントリはシグネチャが0x80であることを考慮しつつ、以下のようなAVX2のコードが書ける。

		const uint64_t hashcode = split_mix_64(entry);
		const uint64_t index = hashcode & ((1ULL << table_bitlen) - 1);
		const uint8_t signature = uint8_t(hashcode >> 57);

		__m256i query_signature = _mm256_set1_epi8(signature);
		__m256i table_signature = _mm256_loadu_si256((__m256i*)&signature_table.data()[index]);

		//[index + i]の情報が ↑のsignature_table.i8[i]に格納されているとして、↓のi桁目のbitに移されるとする。

		const uint64_t is_empty = uint32_t(_mm256_movemask_epi8(table_signature));
		const uint64_t is_positive = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(query_signature, table_signature)));

		uint64_t to_look = (is_empty ^ (is_empty - 1)) & is_positive;

		//[index+i]がシグネチャ陽性かどうかがis_positiveの下からi番目のビットにあるとする。
		//最初に当たる空白エントリより手前にあるシグネチャ陽性な局面の位置のビットボードが計算できる。to_lookがそれである。

		for (uint32_t i = 0; bitscan_forward64(to_look, &i); to_look &= to_look - 1) {
			const uint64_t pos = index + i;
			if (hash_table[pos] == hashcode)return true;
		}
		return false;
	}

	bool get(uint64_t index, uint64_t &dest) {
		if (index >= signature_table.size())return false;
		if (signature_table[index] == 0x80)return false;
		dest = split_mix_64_inverse(hash_table[index]);
		return true;
	}

	void print_all() {
		for (uint64_t i = 0; i < signature_table.size(); ++i) {
			if (signature_table[i] == 0x80)continue;
			const uint64_t code = split_mix_64_inverse(hash_table[i]);
			const std::string s = code_2_unique_string(code);
			std::cout << s << std::endl;
		}
	}
};

template<bool mercy>class OstleEnumerator_search_based {
	
	//mercy: 勝利できる手を見逃さないと辿り着けない局面を列挙するかどうか。

	HashTable searched_position;

	HashTable unsearched_position;

	Moves moves[128];


	void search(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole, const int depth) {

		assert((bb_player & bb_opponent) == 0);
		assert((bb_player | bb_opponent | BB_ALL_8X8_5X5) == BB_ALL_8X8_5X5);
		assert(pos_hole < 25);
		assert(((bb_player | bb_opponent) & (1ULL << pos_2_8x8_5x5_table[pos_hole])) == 0);
		assert(4 <= _mm_popcnt_u64(bb_player) && _mm_popcnt_u64(bb_player) <= 5);
		assert(4 <= _mm_popcnt_u64(bb_opponent) && _mm_popcnt_u64(bb_opponent) <= 5);

		const uint64_t code = code_unique(encode_ostle(bb_player, bb_opponent, pos_hole));

		if (searched_position.find(code)) {
			return;
		}

		if (mercy == false) {
			if (is_checkmate(bb_player, bb_opponent, pos_hole)) {
				searched_position.insert(code);
				return;
			}
		}

		if (depth <= 0) {
			unsearched_position.insert(code);
			return;
		}

		searched_position.insert(code);

		generate_moves(bb_player, bb_opponent, pos_hole, moves[depth]);

		for (int i = moves[depth][0]; i; --i) {
			uint64_t next_bb_player = bb_player, next_bb_opponent = bb_opponent, next_pos_hole = pos_hole;
			do_move(next_bb_player, next_bb_opponent, next_pos_hole, moves[depth][i]);
			if (_mm_popcnt_u64(next_bb_player) == 3)continue;
			if (mercy == true) {
				if (_mm_popcnt_u64(next_bb_opponent) == 3)continue;
			}
			search(next_bb_opponent, next_bb_player, next_pos_hole, depth - 1);
		}
	}

public:

	void do_enumerate() {

		const uint64_t initial_code = code_unique(encode_ostle(0b00011111ULL, 0b00011111'00000000'00000000'00000000'00000000ULL, 12));

		std::vector<uint64_t>task{ initial_code };

		for (uint64_t iteration = 1; task.size() > 0; ++iteration) {

			for (const uint64_t code : task) {
				uint64_t bb_player = 0, bb_opponent = 0, pos_hole = 0;
				decode_ostle(code, bb_player, bb_opponent, pos_hole);
				search(bb_player, bb_opponent, pos_hole, 100);
			}

			task.clear();

			for (uint64_t i = 0; i < unsearched_position.size(); ++i) {
				uint64_t code = 0;
				if (!unsearched_position.get(i, code))continue;
				if (searched_position.find(code))continue;
				task.push_back(code);
			}

			if (task.size() == 0)break;

			unsearched_position.clear();
		}


		std::cout << searched_position.size() << std::endl;

		searched_position.print_all();

		return;
	}
};


class OstleEnumerator_brute_force {

private:

	//HashTable positions;

	std::vector<uint64_t>positions;

	std::chrono::system_clock::time_point t0;


	void position_maker(const uint64_t bb_player, const uint64_t bb_opponent, const uint64_t pos_hole, const int cursor, const int num_piece_player, const int num_piece_opponent) {

		const int num_remaining_object = num_piece_player + num_piece_opponent + (cursor <= pos_hole ? 1 : 0);

		if (cursor == 25) {
			assert(num_remaining_object == 0);
//			const uint64_t code = code_unique(encode_ostle(bb_player, bb_opponent, pos_hole));
			const uint64_t code = encode_ostle(bb_player, bb_opponent, pos_hole);
			positions.push_back(code);

			if (_mm_popcnt_u64(positions.size()) == 1) {
				const auto t1 = std::chrono::system_clock::now();
				const int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
				if (elapsed >= 2000) {
					std::cout << "LOG: positions.size() == " << positions.size() << ", elapsed time = " << elapsed << " milliseconds" << std::endl;
				}
			}
			return;
		}
		else {
			assert(num_remaining_object + cursor < 25 || pos_hole == cursor || num_piece_player + num_piece_opponent > 0);
		}

		if (pos_hole == cursor) {
			position_maker(bb_player, bb_opponent, pos_hole, cursor + 1, num_piece_player, num_piece_opponent);
			return;
		}

		if (num_remaining_object + cursor < 25) {
			position_maker(bb_player, bb_opponent, pos_hole, cursor + 1, num_piece_player, num_piece_opponent);
		}

		const uint64_t bb_cursor = pdep_intrinsics(1ULL << cursor, BB_ALL_8X8_5X5);

		if (num_piece_player > 0) {
			position_maker(bb_player | bb_cursor, bb_opponent, pos_hole, cursor + 1, num_piece_player - 1, num_piece_opponent);
		}
		if (num_piece_opponent > 0) {
			position_maker(bb_player, bb_opponent | bb_cursor, pos_hole, cursor + 1, num_piece_player, num_piece_opponent - 1);
		}
	}

	uint64_t position_maker_root(const uint64_t pos_hole, const int num_piece_player, const int num_piece_opponent) {

		std::cout << "LOG: start: num_unique_positions(" << pos_hole << "," << num_piece_player << "," << num_piece_opponent << ")" << std::endl;

		const auto print_time = [&](std::string task_name) {
			const auto t1 = std::chrono::system_clock::now();
			int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
			std::cout << "LOG: elapsed time:" << task_name << ": " << elapsed << " milliseconds" << std::endl;
			t0 = std::chrono::system_clock::now();
		};

		positions.clear();
		positions.reserve(500'000'000ULL);

		t0 = std::chrono::system_clock::now();

		position_maker(0, 0, pos_hole, 0, num_piece_player, num_piece_opponent);

		{
			const auto t1 = std::chrono::system_clock::now();
			int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
			std::cout << "result: final positions.size() == " << positions.size() << ", elapsed time = " << elapsed << " milliseconds" << std::endl;
		}

		print_time("position_maker");

		const int64_t siz = positions.size();
#pragma omp parallel for
		for (int64_t i = 0; i < siz; ++i) {
			positions[i] = code_unique(positions[i]);
		}

		print_time("fmap code_unique");

		std::sort(std::execution::par, positions.begin(), positions.end());

		print_time("parallel quicksort");

		uint64_t num_unique_positions = 1;
		for (uint64_t i = 1; i < positions.size(); ++i) {
			assert(positions[i - 1] <= positions[i]);
			if (positions[i - 1] != positions[i])++num_unique_positions;
		}

		print_time("counting unique positions");

		std::cout << "result: num_unique_positions(" << pos_hole << "," << num_piece_player << "," << num_piece_opponent << ") = " << num_unique_positions << std::endl;

		return num_unique_positions;
	}

public:

	void do_enumerate() {

		const uint64_t holes[6] = { 0,1,2,6,7,12 };
		uint64_t num_unique_positions = 0;

		for (int i = 0; i < 6; ++i)num_unique_positions += position_maker_root(holes[i], 5, 5);
		for (int i = 0; i < 6; ++i)num_unique_positions += position_maker_root(holes[i], 5, 4);
		for (int i = 0; i < 6; ++i)num_unique_positions += position_maker_root(holes[i], 4, 5);
		for (int i = 0; i < 6; ++i)num_unique_positions += position_maker_root(holes[i], 4, 4);

		std::cout << "result: total number of the unique positions = " << num_unique_positions << std::endl;

		return;
	}

};


int main(int argc, char *argv[]) {

	init_move_tables();

	//test_move(12345, 10000);
	//test_checkmate_detector_func(12345, 10000);
	//test_bitboard_symmetry(12345, 10000);

	//OstleEnumerator_search_based<false> e;
	//e.do_enumerate();

	OstleEnumerator_brute_force e;
	e.do_enumerate();


	return 0;
}

