# opencv_cpu_vs_gpu
A serie of tests to compare performances of **CPU** and **GPU** processing.

This benchmark is based on OpenCV 2.4.13 and performs a simple basic algorithm of computer vision :
* **resize**: to keep costant the process time and indipendent by image size
* **color conversion**: to pass from RGB to grayscale image
* **blur**: to remove noises
* **Canny**: to detect image corners

The algorithm has not a well defined goal, but it may be for example the beginning of a process of "line" or "circle" detection...

The process is iterated "N" times on the same image and a mean of the single step times and of the total time is calculated.

The same algorithm is moved to the **GPU** (only if the machine is *CUDA enabled*).

**GPU TEST**
* *Classic memory copy*: uses the "upload" approach to copy memory from host to device
* *ZERO COPY*: uses "gpu::CudaMem" with "ALLOC_ZEROCOPY" flag to take advantage of shared memory *(if available)*
* *Memory Managed*: allocates memory using "cudaMallocManaged" to take advantage of pinned memory *(if available)*
