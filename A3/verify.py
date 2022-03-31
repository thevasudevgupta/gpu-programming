# python3 verify.py Evaluation_script/testcases/output/output10.txt output.txt

import sys

f1, f2 = sys.argv[1:]
with open(f1) as f1:
    data1 = f1.read().split()

with open(f2) as f2:
    data2 = f2.read().split()

assert data1 == data2, f"{data1}\n{data2}"
# for i in range(len(data1)):
#     if data1[i] != data2[i]:
#         print(f"❌ test case failed at index-{i}")
#         exit()
print("✅ test case passed!!")
