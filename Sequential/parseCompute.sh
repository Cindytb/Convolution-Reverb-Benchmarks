 #!/bin/bash

 cat $1 | grep "Time" | sed 's/Time for CPU convolution: //' | sed 's/ ms//' > computeTemp
 
 python3 parseCompute.py computeTemp $2
 
