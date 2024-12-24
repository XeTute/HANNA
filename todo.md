ANNA-CPU:
- Init
- Forward
- Backward
- batchForward
- batchBackward
- train
- save
- load

ANNA-OCL:
${ANNA-CPU}-OpenCL version

Data-Prep:
- ~loadCSVn~
- ~loadCSVstr~
- saveCSVn
- saveCSVstr

- ~minMaxNorm with overloads~
- ~ZScaleNorm with overloads~