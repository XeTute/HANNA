> ANNA is still new and may contain minor bugs or undefined behavior that we may not have experienced during testing. Please thoroughly test your application before publishing and report any issues that arise.

We're re-designing ANNA to structure the files more logical and the code more efficient. This will also help us port it to OpenCL.

ToDo(see todo.md):
---
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
---
