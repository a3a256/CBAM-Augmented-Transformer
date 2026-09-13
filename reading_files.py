with open("ground_truths/File1.txt") as file:
    lines = [line.rstrip() for line in file]

res = [line.split() for line in lines]

print(res)