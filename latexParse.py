import sys
if len(sys.argv) != 3:
        print("Need 3 arguments: latexParse.py noCols t/f")
        exit(1)
noCols = int(sys.argv[1])
size = sys.argv[2]

num = 0
iteration = 0
w = noCols
h = 0
if(size == "t"):
    h = 25
elif (size == "f"):
    h = 27
A = [ [None]*h for i in range(w) ]
i = 0
while(i<noCols * h):
    A[iteration][num] = input()
    num += 1
    if (num == h):
        num = 0
        iteration += 1
        if (iteration == w):
                break
    i=i+1 

for i in range(h):
    if (h > 9):
        result = "$2^{" + str(i + 4) + "}$ & "
    else:
        result="$2^" + str(i + 4) + "$ & "
    for j in range(iteration - 1):
               result += A[j][i] + " & "
    result += A[iteration - 1][i] + " \\ "
    print(result)
