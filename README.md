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

On the GPU the first benchmark is performed downloading the source image to GPU memory using the default "download" function.
A second benchmark is performed if the GPU device allows the allocation of the memory from host to device using the "**ZERO COPY**" flag, this is the case of Nvidia Jetson boards, for which the benchmark as been mainly written.


