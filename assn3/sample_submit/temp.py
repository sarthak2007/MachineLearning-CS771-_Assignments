import os

cnt = 0
for filename in os.listdir("../train"):
	cnt+=len(filename.split(".")[0])
print(cnt)