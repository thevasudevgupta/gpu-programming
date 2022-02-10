k1 = open("kernel1.txt").read()
k2 = open("kernel2.txt").read()
k3 = open("kernel3.txt").read()
# o = open("sample/Output/large.txt").read()
o = open("output.txt").read()

# print(k1, end="\n")
# print(k2, end="\n")
# print(k3, end="\n")
# print(o, end="\n")

assert k1 == o, "k1 doesn't work"
assert k2 == o, "k2 doesn't work"
assert k3 == o, "k3 doesn't work"
print("Everything works!!!")
