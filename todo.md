ANNA-CPU:
- ~init~
- ~forward~
- ~backward~ replaced with std::vector<float> gradDesc(...)
- batchForward
- batchBackward
- ~train~
- ~save~
- ~load~
- ~display~ // For debugging & analysis

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