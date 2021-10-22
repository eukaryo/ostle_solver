import re
import pathlib
import datetime

from ostle_misc import *

def search_onefile_146move(filename):
    result = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split(",")
            code = decode_base64(row[0])
            assert code is not None

            if len(row) == 2:
                assert row[1] == "checkmate"
                continue
                
            for i in range(1,len(row)):
                if row[i] == "146":
                    digits = convert_code_to_25digits(code)
                    assert digits is not None
                    result.append((digits, i - 1, filename))
    return result

def search_and_print_146move():
    
    print(f"start: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : search_and_print_146move")

    filenames = [str(x) for x in pathlib.Path(".").glob("*.txt")]
    filenames = [x for x in filenames if re.fullmatch(r"ostle_10_retrograde_output[0-9]+\.txt", x) is not None]
    if len(filenames) == 0:
        print("error: ostle_output file is not found in current directory.")
        return

    filenames.sort()
    
    all_result = []
    for i in range(len(filenames)):
        print(f"start: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : {i+1} / {len(filenames)}, filename = {filenames[i]}")
        result = search_onefile_146move(filenames[i])
        all_result += result
        for r in result:
            print(f"found: {r}")

    print(f"finish: num = {len(all_result)}")
    for r in all_result:
        print(f"found: {r}")
        count += 1
    print(f"finish: num = {len(all_result)}")
    
if __name__ == "__main__":

    search_and_print_146move()