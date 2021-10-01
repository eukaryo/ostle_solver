import sys
import re
import typing
import pathlib
import random
import argparse

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
    return min([(code_symmetry(i, code), i) for i in range(1, 8)])

def validate_code(code: int ) -> bool:
    if code is None:
        return False
    if type(code) is not int:
        return False
    if code >> 55 != 0:
        return False
    if code % 32 >= 25:
        return False
    bb1 = (code >> 5) % (1 << 25)
    bb2 = code >> 30
    bbh = 1 << (code % 32)
    if (bb1 & bb2) != 0:
        return False
    if (bb1 & bbh) != 0:
        return False
    if (bb2 & bbh) != 0:
        return False
    pop1 = popcount(bb1)
    if pop1 != 5 and pop1 != 4:
        return False
    pop2 = popcount(bb2)
    if pop2 != 5 and pop2 != 4:
        return False
    return True

def validate_25digits_format(src: str) -> bool:
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

POS_DIFF = (-5,5,-1,1)

def generate_moves(code: int) -> typing.List[typing.Tuple[int, int]]:
    bb_player, bb_opponent, pos_hole = decode_ostle(code)
    answer = []

    b = pext(bb_player, BB_ALL_8X8_5X5)
    while b > 0:
        x = bitscan_forward(b)
        assert x is not None
        answer += [(b,POS_DIFF[i]) for i in range(4)]
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

def do_move(code: int, move: int) -> int:
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
        next_situation = do_move_oneline(situation, x_pos, move_diff)
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
        next_situation = do_move_oneline(situation, x_pos, move_diff // 5)
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
        
def lookup_onefile(filename: str, query_code: int, forbidden_index: int) -> typing.Tuple[bool, int]:
    assert validate_code(query_code)
    query_code = code_unique(query_code)[0]
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()
    code_inside = None
    for line in lines:
        row = line.split(",")
        code = decode_base64(row[0])
        if code is None:
            continue
        if code != query_code:
            code_inside = code
            continue
        
        assert len(row) == 2 or len(row) < forbidden_index + 1
        if len(row) == 2:
            assert row[1] == "checkmate"
            return -1
        assert len(row[forbidden_index + 1]) != 0
        return (True, int(row[forbidden_index + 1]))
    return (False, code_inside)

def determine_forbidden_index(present_code: int, previous_code: typing.Optional[int] = None) -> int:
    assert validate_code(present_code)
    if previous_code is None:
        return 0
    present_code = code_unique(present_code)[0]
    if validate_code(previous_code):
        previous_code = code_unique(previous_code)[0]

    moves = generate_moves(present_code)
    for i in range(len(moves)):
        next_code = do_move(present_code, moves[i])

        next_bb_opponent, next_bb_player, _ = decode_ostle(next_code)
        if popcount(next_bb_opponent) == 3 or popcount(next_bb_player) == 3:
            continue
        assert validate_code(next_code)

        next_code = code_unique(next_code)[0]
        if next_code == previous_code:
            return i + 1
    return 0

def search(present_25digits: str, previous_25digits: typing.Optional[int] = None) -> typing.NoReturn:
    assert validate_25digits_format(present_25digits)
    present_code = convert_25digits_to_code(present_25digits)
    previous_code = None
    if previous_25digits is not None:
        assert validate_25digits_format(previous_25digits)
        previous_code = convert_25digits_to_code(previous_25digits)
    forbidden_index = determine_forbidden_index(present_code, previous_code)

    filenames = [str(x) for x in pathlib.Path(".").glob("*.txt")]
    filenames = [x for x in filenames if re.fullmatch(r"ostle_output[0-9]+\.txt", x) is not None]
    if len(filenames) == 0:
        print("error: ostle_output file is not found in current directory.")
        return
    filenames = [(int(re.search(r"[0-9]+", x)[0]), x) for x in filenames]
    filenames.sort()

    lo, hi = 0, len(filenames)
    while True:
        mid = (lo + hi) // 2
        assert lo + 1 < hi or mid == lo
        (flag, result) = lookup_onefile(filenames[mid][1], present_code, forbidden_index)
        if flag:
            if result == -1:
                print("result: checkmate. Player wins with a minimum of 1 move")
            elif result == 0:
                print("result: draw")
            else:
                assert result > 0
                if result % 2 == 0:
                    print(f"result: player wins with a minimum of {result + 1} moves")
                else:
                    print(f"result: player loses with a maximum of {result + 1} moves")
            return
        else:
            if lo + 1 == hi:
                print("error: query position is not found in the database.")
                return
            if present_code < result:
                hi = mid
            else:
                lo = mid

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
        for j in range(1, 8):
            s = code_symmetry(j, x)
            assert(validate_code(s))
            t = code_symmetry(j, x)
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
    print("unittests: clear!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Ostle solver: output the analytic solution of input state(i.e. present position (and previous position)).')
    parser.add_argument("present", help="present position in the 25digits format")
    parser.add_argument("-p", "--previous", help="previous position in the 25digits format")
    parser.add_argument("-t", "--test", help="run unittests",action="store_true")

    args = parser.parse_args()

    if args.test:
        if unittests(12345,10000) is False:
            sys.exit(1)
    
    if validate_25digits_format(args.present) is False:
        print("error: input (present position) is invalid as the 25digits format")
        sys.exit(1)
    if args.previous is not None:
        if validate_25digits_format(args.previous) is False:
            print("error: input (present position) is invalid as the 25digits format")
            sys.exit(1)
        search(args.present, args.previous)
    else:
        search(args.present)

if __name__ == "__main__":
    main()
