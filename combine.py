import os
import datetime


def solve(n):
    print(f"start: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : processing {n}")
    number = str(n).zfill(4)
    data = []
    for filename in [f"ostle_10_bfs_output{number}.txt", f"ostle_10_retrograde_output{number}.txt"]:
        assert os.path.isfile(filename)
        with open(filename, "r") as f:
            data.append([s.strip().split(",") for s in f.readlines()])
    assert len(data[0]) == len(data[1])
    answer = []
    for i in range(len(data[0])):
        assert data[0][i][0] == data[1][i][0]
        if data[0][i][1] == "checkmate":
            assert data[1][i][1] == "checkmate"
            assert len(data[1][i]) == 2
            assert len(data[0][i]) == 27
            answer.append(",".join(data[0][i]))
        else:
            assert len(data[0][i]) == 26
            assert len(data[1][i]) == 26
            a = [1 if len(x) > 0 else 0 for x in data[0][i]]
            b = [1 if len(x) > 0 else 0 for x in data[1][i]]
            assert sum([abs(a[j] - b[j]) for j in range(26)]) == 0
            suffix = [x for x in data[1][i][1:] if len(x) > 0]
            if len(suffix) == 0: # if unreachable
                answer.append(",".join(data[0][i]))
            else:
                answer.append(",".join(data[0][i])+","+",".join(suffix))
    with open(f"ostle_10_combined{number}.txt", "w") as f:
        for i in range(len(answer)):
            f.write(answer[i] + "\n")


if __name__ == "__main__":

    for i in range(2736):
        solve(i)
    
    print("finish")
