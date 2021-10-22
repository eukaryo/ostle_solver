import re
import pathlib
import datetime

from ostle_misc import *

def find_all_unreachable_position_onefile(filename):
    result = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split(",")
            code = decode_base64(row[0])
            assert code is not None

            start = 1
            if row[1] == "checkmate":
                start = 2

            # bfsの結果が数字のやつだけを集める
            S = [int(x) for x in [row[y] for y in range(start,len(row)) if len(row[y]) > 0]]

            if len(S) == 0: # 盤面row[0]に属する全てのノードが到達不可能な場合（存在する！ 例:00000-00000-00001-11112-22223）
                digits = convert_code_to_25digits(code)
                assert digits is not None
                result.append((digits, filename, start == 2))
    return result

def check_bfs_unreachable_position():
    filenames = [str(x) for x in pathlib.Path(".").glob("*.txt")]
    filenames = [x for x in filenames if re.fullmatch(r"ostle_10_bfs_output[0-9]+\.txt", x) is not None]
    if len(filenames) == 0:
        print("error: ostle_10_bfs_output file is not found in current directory.")
        return

    filenames.sort()
    
    all_result = []
    for i in range(len(filenames)):
        print(f"start: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : {i+1} / {len(filenames)}, filename = {filenames[i]}")
        result = find_all_unreachable_position_onefile(filenames[i])
        all_result += result
        for r in result:
            print(f"found: position = {r[0]}")

    print(f"finish: num = {len(all_result)}")
    for r in all_result:
        print(r)
    print(f"finish: num = {len(all_result)}")

if __name__ == "__main__":

    check_bfs_unreachable_position()
