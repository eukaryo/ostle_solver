import sys
import re
import pathlib
import datetime
import hashlib



if __name__ == "__main__":

    filenames = [str(x) for x in pathlib.Path(".").glob("*.txt")]
    filenames = [x for x in filenames if re.fullmatch(r"ostle_10_(bfs|retrograde)_output[0-9]{4}\.txt", x) is not None]
    filenames.sort()
    answer = ""
    for i in range(len(filenames)):
        print(f"start: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : {i+1} / {len(filenames)}, filename = {filenames[i]}")
        with open(filenames[i], "r") as f:
            filedata = "\n".join([s.rstrip() for s in f.readlines()]) # ignore the difference between LF and CRLF
            hash_sha256 = hashlib.sha256(filedata.encode("utf-8")).hexdigest()
            answer += hash_sha256
    my_digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    correct_digest = "5be55f22f092ecd583965118dc72bf1b2b0949f73cfa03b75f020c18300dd16e"
    print("my digest is:")
    print(my_digest)
    print("correct digest is:")
    print("5be55f22f092ecd583965118dc72bf1b2b0949f73cfa03b75f020c18300dd16e")
    if my_digest == correct_digest:
        print("correct!")
    else:
        print("incorrect...")
