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
- CSV.hpp:
- - ~loadCSVn~
- - ~loadCSVstr~
- - ~saveCSV~

- normalize.hpp:
- - ~minMaxNorm with overloads~
- - ~ZScaleNorm with overloads~