import sys
if len(sys.argv) != 3:
	print("Need 3 arguments: parse.py input.txt output.txt")
	exit(1)
inFile = sys.argv[1]
outFile = sys.argv[2]
file = open(inFile, 'r')


num = 0
iteration = 0

w,h = 10, 27
A = [ [None]*h for i in range(w) ]
otherFile = open(outFile, 'w')
for l in file:
	if l.isspace():
		iteration += 1
		num = 0
		continue
	else:
		m, s = l.split("m")
		m = int(m)
		s = float(s)
		A[iteration][num] = str(m * 60 + s)
		num += 1
result="Size of input,"
for i in range (iteration - 1):
	
	result += "Iteration "+ str(i) + ","
result += "Iteration " + str(iteration - 1)
otherFile.write(result + "\n")
num = h
iteration = 10
for i in range(num):
	result="2^" + str(i + 4) + ","
	for j in range(iteration - 1):
		result += A[j][i] + ","
	result += A[iteration - 1][i]
	otherFile.write(result + "\n")
file.close()
otherFile.close()
