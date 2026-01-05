
# Hardware Accelerator for Gemma 3 LLM on FPGA 

> **Note:** This project is currently a Work in Progress. The core compute kernel and system communication interfaces are completed. Host software integration and user interface components will be updated in future releases.


![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

LLM inference in C/C++
## Introduction: 
This project focuses on designing and executing a hardware accelerator  for the  ** Gemma 3 LLM ** Quantized Int 8 on the **Xilinx Kria KV260 FPGA**. 

The primary objective is to solve the "Memory Wall" bottleneck and enhance inference speed on edge devices. This is achieved by offloading heavy matrix multiplication operations (GEMM/GEMV) from the CPU to the FPGA using a custom compute kernel optimized.

# Feature: 
* ** Quantization IN8 model Support: Perform computation driectly on 8 bit quantized, compatiable with GGUF format ( you can do this project with other model like: Llama, Qwen, ...), reducing memory bandwidth requirement by 4x comparing with Float 32 bit.
* Split array Memory layout: Implements a separaating scale factors and weights to resolve struct misalignment issues, optimizing data alignment for AXI 4 Burst.
* The kernel forward IP:
   Decompse large matrix into fixed 16x64 blocks to fit on chip BRAM resources

   Utilized buffering to paralleleze load, compute, and store.

  Leverages `Loop Unrolling` and `Cyclic Array Partitioning` to execute 16 MAC operations per clock cycle, maximizing DSP utilization.
* System integration: SOC dessign on Zynq Ultrascale + MPsoc, utilizing AXI4 lite for control and  4 paralell AXI4-Full port throught Smartconnect for high bandwidth data transfrer.
## SOC system 

### Block dessign on Vivado: 

<img width="1066" height="587" alt="image" src="https://github.com/user-attachments/assets/ce7e0c91-3d5e-44cc-80ad-4b8e3fd57ef2" />

### Data flow: 

<img width="828" height="724" alt="image" src="https://github.com/user-attachments/assets/a740a0a8-f86f-4e75-acf2-74f23b95efe1" />

   
        
