> ANNA is still new and may contain minor bugs or undefined behavior that we may not have experienced during testing. Please thoroughly test your application before publishing and report any issues that arise.

# ANNA  
ANNA stands for **Asadullah's Neural Network Architecture**. This repository contains the implementation of ANNA, which is why we named both the repository and the library "ANNA."  
The naming might be a bit confusing at first, but once you get used to it, navigating will be straightforward.

# To-Do
- 🔴 Add a function _ANNA::ANNA::inference which forward-props a batch of input matracies
- 🟢 ~In _ANNA::ANNA::train, also train on the modulo samples (`counter chunkRemainder = d[0].size() % chunkSize;`)~
- 🔴 Upload OpenCL beta, which may include bugs and create a note at init for it (`[ANNA OpenCL-Version]: The OCL Version of ANNA is still in the testing phase. Please test before production deployment.`)

# What This Repository Is  
This repository is a **header-only library** that requires no dependencies for CPU usage and uses **OpenCL** for GPU support.  
Initially, we planned to implement support only for CPUs and CUDA, but Intel's latest GPUs have proven excellent for consumer AI tasks. Additionally, OpenCL is not only compatible with GPUs but also runs on other hardware, making it a smarter choice overall.

# Language  
ANNA is a native **C++ header-only library**. We currently have no plans to port it to other languages, such as Python or Java.

# Learn How to Use ANNA  
You can read the **Wiki** in this repository. While it isn't updated very frequently, it still serves as a great resource, and as soon as we upload or remove a function, we'll add it there quickly enough.<br>
Apart from just reading Wikis, you can also check out our [YouTube Channel](https://www.youtube.com/@XeTuteTechnologies) for Tutorials.

# Happy Coding!  

# Star History  
<a href="https://star-history.com/#XeTute/ANNA&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=XeTute/ANNA&type=Date" />
 </picture>
</a>
