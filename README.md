
# Hardware Accelerator for Gemma 3 LLM on FPGA 

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

LLM inference in C/C++
## Introduction: 
This project focuses on designing and executing a hardware accelerator  for the  ** Gemma 3 LLM ** Quantized Int 8 on the **Xilinx Kria KV260 FPGA**. 

The primary objective is to solve the "Memory Wall" bottleneck and enhance inference speed on edge devices. This is achieved by offloading heavy matrix multiplication operations (GEMM/GEMV) from the CPU to the FPGA using a custom compute kernel optimized.
> **Note:** This project is currently a Work in Progress (WIP). The core compute kernel and system communication interfaces are complete. Host software integration and user interface components will be updated in future releases.
