import sys
import re
import typing
import pathlib
import random
import argparse
import datetime

verbose_flag = False

def pdep(a: int, mask: int) -> int:
    dest = 0
    k = 0
    for m in range(64):
        if (mask & (1 << m)) != 0:
            if (a & (1 << k)) != 0:
                dest += 1 << m
            k += 1
    return dest

def pext(a: int, mask: int) -> int:
    dest = 0
    k = 0
    for m in range(64):
        if (mask & (1 << m)) != 0:
            if (a & (1 << m)) != 0:
                dest += (1 << k)
            k += 1
    return dest

def popcount(n: int) -> int:
    return bin(n).count("1")

def bitscan_forward(n: int) -> typing.Optional[int]:
    if n <= 0:
        return None
    count = 0
    while n % 2 == 0:
        count += 1
        n = n // 2
    return count

BASE64CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

def encode_base64(src: int) -> str:
    answer = ""
    while src > 0:
        answer += BASE64CHARACTERS[src % 64]
        src //= 64
    return answer

def decode_base64(src: str) -> typing.Optional[int]:
    if len(src) == 0:
        return None
    array = [BASE64CHARACTERS.find(src[i]) for i in range(len(src))]
    if min(array) == -1:
        return None
    result = sum([array[i] * (64 ** i) for i in range(len(src))])
    if result >= 2 ** 55:
        return None
    return result

BB_ALL_8X8_5X5 = 0b0001111100011111000111110001111100011111

def encode_ostle(bb_player: int, bb_opponent: int, pos_hole: int) -> int:
    code = pos_hole
    code += pext(bb_player, BB_ALL_8X8_5X5) << 30
    code += pext(bb_opponent, BB_ALL_8X8_5X5) << 5
    return code

def decode_ostle(code: int) -> typing.Tuple[int, int, int]:
    assert code < 2 ** 55
    pos_hole = code % 32
    bb_player = pdep(code >> 30, BB_ALL_8X8_5X5)
    bb_opponent = pdep(code >> 5, BB_ALL_8X8_5X5)
    return (bb_player, bb_opponent, pos_hole)

def transpose_5x5_bitboard(b: int) -> int:
    t = (b ^ (b >> 7)) & 0x00aa00aa00aa00aa
    b = b ^ t ^ (t << 7)
    t = (b ^ (b >> 14)) & 0x0000cccc0000cccc
    b = b ^ t ^ (t << 14)
    t = (b ^ (b >> 28)) & 0x00000000f0f0f0f0
    b = b ^ t ^ (t << 28)
    return b

def vertical_mirror_5x5_bitboard(b: int) -> int:
    b = ((b >> 8) & 0x00FF00FF00FF00FF) | ((b << 8) & 0xFF00FF00FF00FF00)
    b = ((b >> 16) & 0x0000FFFF0000FFFF) | ((b << 16) & 0xFFFF0000FFFF0000)
    b = ((b >> 32) & 0x00000000FFFFFFFF) | ((b << 32) & 0xFFFFFFFF00000000)
    return b >> 24

def horizontal_mirror_5x5_bitboard(b: int) -> int:
    b = ((b >> 1) & 0x5555555555555555) | ((b << 1) & 0xAAAAAAAAAAAAAAAA)
    b = ((b >> 2) & 0x3333333333333333) | ((b << 2) & 0xCCCCCCCCCCCCCCCC)
    b = ((b >> 4) & 0x0F0F0F0F0F0F0F0F) | ((b << 4) & 0xF0F0F0F0F0F0F0F0)
    return b >> 3

transpose_5x5_table = (0,5,10,15,20,1,6,11,16,21,2,7,12,17,22,3,8,13,18,23,4,9,14,19,24)
vertical_mirror_5x5_table = (20,21,22,23,24,15,16,17,18,19,10,11,12,13,14,5,6,7,8,9,0,1,2,3,4)
horizontal_mirror_5x5_table = (4,3,2,1,0,9,8,7,6,5,14,13,12,11,10,19,18,17,16,15,24,23,22,21,20)

def code_symmetry(s: int, code: int) -> int:
    bb1, bb2, pos = decode_ostle(code)
    if (s & 1) != 0:
        bb1 = horizontal_mirror_5x5_bitboard(bb1)
        bb2 = horizontal_mirror_5x5_bitboard(bb2)
        pos = horizontal_mirror_5x5_table[pos]
    if (s & 2) != 0:
        bb1 = vertical_mirror_5x5_bitboard(bb1)
        bb2 = vertical_mirror_5x5_bitboard(bb2)
        pos = vertical_mirror_5x5_table[pos]
    if (s & 4) != 0:
        bb1 = transpose_5x5_bitboard(bb1)
        bb2 = transpose_5x5_bitboard(bb2)
        pos = transpose_5x5_table[pos]
    return encode_ostle(bb1, bb2, pos)

def code_unique(code: int) -> int:
    return min([code_symmetry(i, code) for i in range(8)])

def validate_code(code: int ) -> bool:
    if code is None:
        #print("error: validate_code: code is None")
        return False
    if type(code) is not int:
        #print("error: validate_code: type(code) is not int")
        return False
    if code >> 55 != 0:
        #print("error: validate_code: code >> 55 != 0")
        return False
    if code % 32 >= 25:
        #print("error: validate_code: code % 32 >= 25")
        return False
    bb1 = (code >> 5) % (1 << 25)
    bb2 = code >> 30
    bbh = 1 << (code % 32)
    if (bb1 & bb2) != 0:
        #print("error: validate_code: (bb1 & bb2) != 0")
        return False
    if (bb1 & bbh) != 0:
        #print("error: validate_code: (bb1 & bbh) != 0")
        return False
    if (bb2 & bbh) != 0:
        #print("error: validate_code: (bb2 & bbh) != 0")
        return False
    pop1 = popcount(bb1)
    if pop1 != 5 and pop1 != 4:
        #print("error: validate_code: pop1 != 5 and pop1 != 4")
        return False
    pop2 = popcount(bb2)
    if pop2 != 5 and pop2 != 4:
        #print("error: validate_code: pop1 != 5 and pop1 != 4")
        return False
    return True

def validate_25digits_format(src: str) -> bool:
    src = src.replace("-","")
    if re.fullmatch(r"[0123]{25}", src) is None:
        return False
    if src.count("3") != 1:
        return False
    if src.count("2") != 5 and src.count("2") != 4:
        return False
    if src.count("1") != 5 and src.count("1") != 4:
        return False
    return True

def convert_25digits_to_code(src: str) -> typing.Optional[int]:
    src = src.replace("-","")
    if validate_25digits_format(src) is False:
        print("error: convert_25digits_to_code: invalid position")
        return None
    bb_player, bb_opponent, pos_hole = 0, 0, 0
    for i in range(5):
        for j in range(5):
            if src[i * 5 + j] == "1":
                bb_player += 1 << (i * 8 + j)
            elif src[i * 5 + j] == "2":
                bb_opponent += 1 << (i * 8 + j)
            elif src[i * 5 + j] == "3":
                pos_hole = (i * 5 + j)
    answer = encode_ostle(bb_player, bb_opponent, pos_hole)
    assert validate_code(answer)
    return answer

def convert_code_to_25digits(src: int) -> typing.Optional[str]:
    if validate_code(src) is False:
        print("error: convert_code_to_25digits: invalid position")
        return None
    bb_player, bb_opponent, pos_hole = decode_ostle(src)
    answer = ""
    for i in range(5):
        for j in range(5):
            if (bb_player & (1 << (i * 8 + j))) != 0:
                answer += "1"
            elif (bb_opponent & (1 << (i * 8 + j))) != 0:
                answer += "2"
            elif pos_hole == (i * 5 + j):
                answer += "3"
            else:
                answer += "0"
    assert validate_25digits_format(answer)
    return answer

def convert_code_to_25digits_unsafe(src):
    bb_player, bb_opponent, pos_hole = decode_ostle(src)
    answer = ""
    for i in range(5):
        for j in range(5):
            if (bb_player & (1 << (i * 8 + j))) != 0:
                answer += "1"
            elif (bb_opponent & (1 << (i * 8 + j))) != 0:
                answer += "2"
            elif pos_hole == (i * 5 + j):
                answer += "3"
            else:
                answer += "0"
    return answer

def insert_hyphen_to_25digits(src):
    if validate_25digits_format(src) is False:
        print("error: insert_hyphen_to_25digits: invalid position")
        return None
    src = src.replace("-","")
    return src[0:5]+"-"+src[5:10]+"-"+src[10:15]+"-"+src[15:20]+"-"+src[20:25]

def insert_hyphen_to_25digits_unsafe(src):
    src = src.replace("-","")
    return src[0:5]+"-"+src[5:10]+"-"+src[10:15]+"-"+src[15:20]+"-"+src[20:25]

POS_DIFF = (-5,5,-1,1)

def generate_moves(code: int) -> typing.List[typing.Tuple[int, int]]:
    bb_player, bb_opponent, pos_hole = decode_ostle(code)
    answer = []

    b = pext(bb_player, BB_ALL_8X8_5X5)
    while b > 0:
        x = bitscan_forward(b)
        assert x is not None
        answer += [(x,POS_DIFF[i]) for i in range(4)]
        b &= b - 1

    bb_empty = BB_ALL_8X8_5X5 ^ (bb_player | bb_opponent)
    bb_hole = pdep(2 ** pos_hole, BB_ALL_8X8_5X5)
    if ((bb_hole >> 8) & bb_empty) != 0:
        answer.append((pos_hole, POS_DIFF[0]))
    if ((bb_hole << 8) & bb_empty) != 0:
        answer.append((pos_hole, POS_DIFF[1]))
    if ((bb_hole >> 1) & bb_empty) != 0:
        answer.append((pos_hole, POS_DIFF[2]))
    if ((bb_hole << 1) & bb_empty) != 0:
        answer.append((pos_hole, POS_DIFF[3]))

    return answer

def do_move_oneline(situation: list, move: typing.Tuple[int, int]) -> typing.List[str]:
    move_index, direction = move
    assert len(situation) == 5
    assert 0 <= move_index and move_index < 5
    assert abs(direction) == 1
    assert situation[move_index] == "PLAYER" or situation[move_index] == "HOLE"

    pushing_piece = situation[move_index]
    situation[move_index] = "EMPTY"
    while True:
        next_move_index = move_index + direction
        if next_move_index < 0 or 5 <= next_move_index or situation[next_move_index] == "HOLE":
            break
        if situation[next_move_index] == "EMPTY":
            situation[next_move_index] = pushing_piece
            break
        pushing_piece, situation[next_move_index] = situation[next_move_index], pushing_piece
        move_index = next_move_index

    return situation

def do_move(code: int, move: typing.Tuple[int, int]) -> int:
    bb_player, bb_opponent, pos_hole = list(decode_ostle(code))
    move_pos, move_diff = move

    if pos_hole == move_pos:
        return encode_ostle(bb_opponent, bb_player, pos_hole + move_diff)
    
    x_pos = move_pos % 5
    y_pos = move_pos // 5
    situation = []
    if abs(move_diff) == 1: # horizontal
        for i in range(5):
            if (bb_player & (1 << (y_pos * 8 + i))) != 0:
                situation.append("PLAYER")
            elif (bb_opponent & (1 << (y_pos * 8 + i))) != 0:
                situation.append("OPPONENT")
            elif pos_hole == y_pos * 5 + i:
                situation.append("HOLE")
            else:
                situation.append("EMPTY")
        next_situation = do_move_oneline(situation, (x_pos, move_diff))
        bb_mask = BB_ALL_8X8_5X5 ^ (0b11111 << (y_pos * 8))
        bb_player &= bb_mask
        bb_opponent &= bb_mask
        for i in range(5):
            if next_situation[i] == "PLAYER":
                bb_player |= 1 << (y_pos * 8 + i)
            elif next_situation[i] == "OPPONENT":
                bb_opponent |= 1 << (y_pos * 8 + i)
    elif abs(move_diff) == 5: # vertical
        for i in range(5):
            if (bb_player & (1 << (i * 8 + x_pos))) != 0:
                situation.append("PLAYER")
            elif (bb_opponent & (1 << (i * 8 + x_pos))) != 0:
                situation.append("OPPONENT")
            elif pos_hole == i * 5 + x_pos:
                situation.append("HOLE")
            else:
                situation.append("EMPTY")
        next_situation = do_move_oneline(situation, (y_pos, move_diff // 5))
        bb_mask = BB_ALL_8X8_5X5 ^ ((1 + (1 << 8) + (1 << 16) + (1 << 24) + (1 << 32)) << x_pos)
        bb_player &= bb_mask
        bb_opponent &= bb_mask
        for i in range(5):
            if next_situation[i] == "PLAYER":
                bb_player |= 1 << (i * 8 + x_pos)
            elif next_situation[i] == "OPPONENT":
                bb_opponent |= 1 << (i * 8 + x_pos)
    else:
        assert False
    return encode_ostle(bb_opponent, bb_player, pos_hole)

def determine_forbidden_index(present_code: int, previous_code: typing.Optional[int] = None) -> int:
    assert validate_code(present_code)
    if previous_code is None:
        return 0
    present_code = code_unique(present_code)
    if validate_code(previous_code):
        previous_code = code_unique(previous_code)

    moves = generate_moves(present_code)
    for i in range(len(moves)):
        next_code = do_move(present_code, moves[i])

        next_bb_opponent, next_bb_player, _ = decode_ostle(next_code)
        if popcount(next_bb_opponent) == 3 or popcount(next_bb_player) == 3:
            continue
        assert validate_code(next_code)

        next_code = code_unique(next_code)
        if next_code == previous_code:
            return i + 1
    return 0

def make_random_position_code(seed):
    state = random.getstate()
    random.seed(seed)
    pos_hole = random.randrange(25)
    bb1, bb2, bbp = 0, 0, 1 << pos_hole
    pop1 = random.randrange(4,6)
    pop2 = random.randrange(4,6)
    while popcount(bb1) != pop1:
        pos = random.randrange(25)
        if ((bb1 | bb2 | bbp) & (1 << pos)) == 0:
            bb1 += 1 << pos
    while popcount(bb2) != pop2:
        pos = random.randrange(25)
        if ((bb1 | bb2 | bbp) & (1 << pos)) == 0:
            bb2 += 1 << pos
    random.setstate(state)
    answer = pos_hole + (bb1 << 5) + (bb2 << 30)
    assert validate_code(answer)
    return answer

def unittests(seed, num_iter: int) -> bool:
    random.seed(seed)
    for i in range(num_iter):
        if i % (num_iter // 10) == 0:
            print(f"unittests: {i} / {num_iter}") 
        x = make_random_position_code(random.randrange(2 ** 55))
        assert(validate_code(x))
        for j in [1,2,4]:
            s = code_symmetry(j, x)
            assert(validate_code(s))
            t = code_symmetry(j, s)
            assert x == t
        y1 = encode_base64(x)
        z1 = decode_base64(y1)
        if x != z1:
            print(x)
            print(y1)
            print(z1)
            print("test failed: encode_base64 or decode_base64")
            return False
        (bb1, bb2, pos) = decode_ostle(x)
        z2 = encode_ostle(bb1, bb2, pos)
        if x != z2:
            print(x)
            print((bb1, bb2, pos))
            print(z2)
            print("test failed: decode_ostle or encode_ostle")
            return False
        y3 = convert_code_to_25digits(x)
        z3 = convert_25digits_to_code(y3)
        if x != z3:
            print(x)
            print(y3)
            print(z3)
            print("test failed: convert_code_to_25digits or convert_25digits_to_code")
            return False
    print("unittests: clear!")
    return True

if __name__ == "__main__":
    unittests(12345, 10000)
