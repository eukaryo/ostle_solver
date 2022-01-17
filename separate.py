import os
import datetime


def solve(n):
    print(f"start: {datetime.datetime.now().strftime(r'%Y/%m/%d %H:%M:%S')} : processing {n}")
    number = str(n).zfill(4)
    filename = f"ostle_10_combined{number}.txt"
    assert os.path.isfile(filename)
    with open(filename, "r") as f:
        data = [s.strip().split(",") for s in f.readlines()]
    answer_bfs = []
    answer_retrograde = []
    for i in range(len(data)):
        if data[i][1] == "checkmate":
            assert len(data[i]) == 27
            answer_bfs.append(data[i])
            answer_retrograde.append(data[i][:2])
        elif sum([len(data[i][j]) for j in range(1,26)]) == 0: # unreachable
            answer_bfs.append(data[i][:26])
            answer_retrograde.append(data[i][:26])
        else:
            answer_bfs.append(data[i][:26])
            answer_retrograde.append(data[i][:26])
            count = 26
            for j in range(1,26):
                if len(answer_retrograde[-1][j]) > 0:
                    answer_retrograde[-1][j] = data[i][count]
                    count += 1
            assert count == len(data[i])
    with open(f"ostle_10_bfs_output{number}.txt", "w") as f:
        for i in range(len(answer_bfs)):
            f.write(",".join(answer_bfs[i]) + "\n")
    with open(f"ostle_10_retrograde_output{number}.txt", "w") as f:
        for i in range(len(answer_retrograde)):
            f.write(",".join(answer_retrograde[i]) + "\n")


if __name__ == "__main__":

    for i in range(2736):
        solve(i)
    
    print("finish")
