import sys
import re
import typing
import pathlib
import argparse

from ostle_misc import *

verbose_flag = False

def lookup_oneline_bfs(line: str, query_code: int, forbidden_index: int, filename: str) -> typing.Optional[typing.Tuple[bool, int]]:
    query_code = code_unique(query_code)
    row = line.strip().split(",")
    code_found = decode_base64(row[0])
    if code_found is None:
        return None
    if code_found != query_code:
        return (False, code_found)
    elif verbose_flag:
        print("LOG: lookup_oneline_bfs: present position is found!")

    assert len(row) >= 2

    start = 1
    if row[1] == "checkmate":
        start = 2

    assert forbidden_index + start < len(row)

    if len(row[forbidden_index + start]) == 0:
        print(f"error: lookup_oneline_bfs: forbidden_index = {forbidden_index} is invalid in the position {query_code}. filename = {filename}")
        sys.exit(0)
    return (True, int(row[forbidden_index + start]))

def lookup_oneline_retrograde(line: str, query_code: int, forbidden_index: int, filename: str) -> typing.Optional[typing.Tuple[bool, int]]:
    query_code = code_unique(query_code)
    row = line.strip().split(",")
    code_found = decode_base64(row[0])
    if code_found is None:
        return None
    if code_found != query_code:
        return (False, code_found)
    elif verbose_flag:
        print("LOG: lookup_oneline_retrograde: present position is found!")

    assert len(row) == 2 or forbidden_index + 1 < len(row)
    if len(row) == 2:
        assert row[1] == "checkmate"
        return (True, -1)
    if len(row[forbidden_index + 1]) == 0:
        print(f"error: lookup_oneline_retrograde: forbidden_index = {forbidden_index} is invalid in the position {query_code}. filename = {filename}")
        sys.exit(0)
    return (True, int(row[forbidden_index + 1]))

def lookup_onefile_topline_only_retrograde(filename: str, query_code: int, forbidden_index: int) -> typing.Optional[typing.Tuple[bool, int]]:
    assert validate_code(query_code)
    query_code = code_unique(query_code)
    with open(filename, "r") as f:
        line = f.readline()
        if len(line) == 0:
            print(f"error: lookup_onefile_topline_only_retrograde: {filename} does not contain any position.")
            return None
        
        x = lookup_oneline_retrograde(line, query_code, forbidden_index, filename)
        if x is None:
            print(f"error: lookup_onefile_topline_only_retrograde: the first line of {filename} is invalid.")
        return (x[0], x[1], 0)

    print(f"error: lookup_onefile_topline_only_retrograde: failed to lookup {filename}.")
    return None

def lookup_onefile_binarysearch_retrograde(filename: str, query_code: int, forbidden_index: int):
    assert validate_code(query_code)
    query_code = code_unique(query_code)
    lines = None
    with open(filename, "r") as f:
        lines = f.readlines()
    if lines is None:
        print(f"error: lookup_onefile_binarysearch_retrograde: failed to lookup {filename}.")

    x = lookup_oneline_retrograde(lines[0], query_code, forbidden_index, filename)
    if x is None:
        print(f"error: lookup_onefile_binarysearch_retrograde: the first line of {filename} is invalid.")
    if x[0]:
        return (x[0], x[1], 0)
    if query_code < x[1]:
        return (x[0], x[1], None)
    
    x = lookup_oneline_retrograde(lines[-1], query_code, forbidden_index, filename)
    if x is None:
        print(f"error: lookup_onefile_binarysearch_retrograde: the last line of {filename} is invalid.")
    if x[0]:
        return (x[0], x[1], len(lines) - 1)
    if x[1] < query_code:
        return (x[0], x[1], None)
    
    lo, hi = 0, len(lines)
    while True:
        mid = (lo + hi) // 2
        assert lo + 1 < hi or mid == lo
        x = lookup_oneline_retrograde(lines[mid], query_code, forbidden_index, filename)
        if x is None:
            print(f"error: lookup_onefile_binarysearch_retrograde: the {mid+1}-th line of {filename} is invalid.")
        if x[0]:
            return (x[0], x[1], mid)
        else:
            if lo + 1 == hi:
                print("error: lookup_onefile_binarysearch_retrograde: binarysearch is failed.")
                return None
            if query_code < x[1]:
                hi = mid
            else:
                lo = mid

def print_result_bfs(x: int, filename):
    
    assert x >= 0
    if verbose_flag:
        print(f"result(bfs): filename = {filename}")
    if x == 0:
        print("result(bfs): the initial state.")
    else:
        print(f"result(bfs): it takes at least {x} move(s) from the initial state.")

def print_result_retrograde(x: int, filename):
    if verbose_flag:
        print(f"result(retrograde): filename = {filename}")
    if x == -1:
        print("result(retrograde): checkmate. Player wins in one move.")
    elif x == 0:
        print("result(retrograde): draw.")
    else:
        assert x > 0
        if x % 2 == 0:
            print(f"result(retrograde): player wins in {x + 1} moves.")
        else:
            print(f"result(retrograde): player loses in {x + 1} moves.")

def decode_digits(present_25digits, previous_25digits = None):
    assert validate_25digits_format(present_25digits)
    present_code = convert_25digits_to_code(present_25digits)
    previous_code = None
    if previous_25digits is not None:
        assert validate_25digits_format(previous_25digits)
        previous_code = convert_25digits_to_code(previous_25digits)
    forbidden_index = determine_forbidden_index(present_code, previous_code)
    return (present_code, previous_code, forbidden_index)

def search_retrograde(present_25digits, previous_25digits = None):
    
    (present_code, previous_code, forbidden_index) = decode_digits(present_25digits, previous_25digits)

    filenames = [str(x) for x in pathlib.Path(".").glob("*.txt")]
    filenames = [x for x in filenames if re.fullmatch(r"ostle_10_retrograde_output[0-9]+\.txt", x) is not None]
    if len(filenames) == 0:
        print("error: ostle_10_retrograde_output file is not found in current directory.")
        return

    filenames.sort()

    lo, hi = 0, len(filenames)
    while True:
        mid = (lo + hi) // 2
        if verbose_flag:
            print(f"LOG:(lo,hi,mid)=({lo},{hi},{mid})")
        assert lo + 1 < hi or mid == lo
        x = lookup_onefile_topline_only_retrograde(filenames[mid], present_code, forbidden_index)
        if verbose_flag:
            print(f"LOG:x={x}")
        if x is None:
            print(f"error: search: {filenames[mid]} is invalid.")
        if x[0]:
            return {"distance":x[1],"filename":filenames[mid],"line":x[2]}
        else:
            if lo + 1 == hi:
                break
            if code_unique(present_code) < x[1]:
                hi = mid
            else:
                lo = mid
    if verbose_flag:
        print(f"LOG:(lo,hi)=({lo},{hi})")
    x = lookup_onefile_binarysearch_retrograde(filenames[lo], present_code, forbidden_index)
    if verbose_flag:
        print(f"LOG:x={x}")
    if x is None:
        print(f"error: search: {filenames[lo]} is invalid.")
    elif x[0] is False:
        print(f"error: search: file-wise binarysearch is failed.")
    else:
        return {"solution":x[1],"filename":filenames[lo],"line_number":x[2]}
    return None

def is_suicided(code: int ) -> bool:
    bb1 = (code >> 5) % (1 << 25)
    pop1 = popcount(bb1)
    if pop1 == 3:
        return True
    return False

def search_sequence(present_25digits, previous_25digits = None):

    present_25digits = present_25digits.replace("-","")
    if previous_25digits is not None:
        previous_25digits = previous_25digits.replace("-","")

    print("start: search_sequence")

    while True:
        (present_code, previous_code, _) = decode_digits(present_25digits, previous_25digits)
        x = search_retrograde(present_25digits, previous_25digits)
        
        print(f"present_25digits={insert_hyphen_to_25digits(present_25digits)},previous_25digits={insert_hyphen_to_25digits(previous_25digits) if previous_25digits is not None else 'None'},solution={x['solution']}")

        if x["solution"] == -1 or x["solution"] == 0:
            break
        moves = generate_moves(present_code)
        found_flag = False
        for i in range(len(moves)):
            next_code = do_move(present_code, moves[i])

            next_25digits = convert_code_to_25digits_unsafe(next_code)

            if is_suicided(next_code):
                print(f"i={str(i).rjust(2,' ')},next_25digits={insert_hyphen_to_25digits_unsafe(next_25digits)},suicide move")
                continue

            assert validate_code(next_code)

            y = search_retrograde(next_25digits, present_25digits)

            if next_25digits == previous_25digits:
                print(f"i={str(i).rjust(2,' ')},next_25digits={insert_hyphen_to_25digits(next_25digits)},y={y} next_25digits == previous_25digits")
                continue
            else:
                print(f"i={str(i).rjust(2,' ')},next_25digits={insert_hyphen_to_25digits(next_25digits)},y={y}")

            if x["solution"] == 1:
                assert y["solution"] == -1
            elif x["solution"] % 2 == 1:
                assert y["solution"] == -1 or (y["solution"] >= 2 and y["solution"] % 2 == 0 and y["solution"] <= x["solution"] - 1)
            elif x["solution"] % 2 == 0:
                assert y["solution"] == -1 or y["solution"] % 2 == 0 or (y["solution"] % 2 == 1 and y["solution"] >= x["solution"] - 1)
            else:
                assert False

            if y["solution"] == x["solution"] - 1 or (y["solution"] == -1 and x["solution"] == 1):
                if found_flag is False:
                    present_25digits_, previous_25digits_ = next_25digits, present_25digits
                    found_flag = True
        assert found_flag
        present_25digits, previous_25digits = present_25digits_, previous_25digits_

def solve(sequence_flag, present_25digits, previous_25digits = None):

    (present_code, previous_code, forbidden_index) = decode_digits(present_25digits, previous_25digits)

    if verbose_flag:
        print(f"present_code={present_code}")
        print(f"code_unique(present_code)={code_unique(present_code)}")
        print(f"encode_base64(code_unique(present_code))={encode_base64(code_unique(present_code))}")
        print(f"forbidden_index={forbidden_index}")

    x = search_retrograde(present_25digits, previous_25digits)

    if x is None:
        sys.exit(0)

    print_result_retrograde(x["solution"], x["filename"])

    filename = x["filename"].replace("retrograde", "bfs")
    with open(filename, "r") as f:
        lines = f.readlines()
        y = lookup_oneline_bfs(lines[x["line_number"]], present_code, forbidden_index, filename)
        assert y[0]
        print_result_bfs(y[1], filename)

    if sequence_flag:
        search_sequence(present_25digits, previous_25digits)

def main():
    parser = argparse.ArgumentParser(description='Ostle solver: output the analytic solution of input state(i.e. present position (and previous position)).')
    parser.add_argument("present", help="present position in the 25digits format")
    parser.add_argument("-p", "--previous", help="previous position in the 25digits format")
    parser.add_argument("-t", "--test", help="run unittests",action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",action="store_true")
    parser.add_argument("-s", "--sequence", help="if not draw, print sequence to checkmate and all alternative moves",action="store_true")

    args = parser.parse_args()

    if args.test:
        if unittests(12345,10000) is False:
            sys.exit(0)
    
    if args.verbose:
        global verbose_flag
        verbose_flag = True
    
    if validate_25digits_format(args.present) is False:
        print("error: input (present position) is invalid as the 25digits format")
        sys.exit(0)
    if args.previous is not None:
        if validate_25digits_format(args.previous) is False:
            print("error: input (present position) is invalid as the 25digits format")
            sys.exit(0)
        solve(args.sequence, args.present, args.previous)
    else:
        solve(args.sequence, args.present)

if __name__ == "__main__":
    main()
