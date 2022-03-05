target_path = "evaluation-script/testcases/output/output3.txt"
result_path = "output.txt"

with open(target_path) as f:
    target = f.read()

with open(result_path) as f:
    result = f.read()

# print(target.split())
# print(result.split())

assert target.split() == result.split(), f"\ntarget: {target}\nresult: {result}"
print("SUCCESSFUL !!")
