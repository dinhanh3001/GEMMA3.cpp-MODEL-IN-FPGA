#ifndef FPGA_HOST_H
#define FPGA_HOST_H

#include <string>
#include <cstddef>

// Initialize FPGA host. If successful, return true. Otherwise set
// `err` to a human-readable message and return false.
bool fpga_host_init(const std::string &xclbin_path, const std::string &kernel_name, std::string &err);

// Shutdown / cleanup
void fpga_host_shutdown();

// Returns true if FPGA layer is initialized and ready.
bool fpga_ready();

// Allocate a BO on the device. Returns index >= 0 on success, -1 on failure.
int fpga_alloc_bo(size_t bytes, int bank = 0);

// Write to BO at index
bool fpga_bo_write(int idx, const void * src, size_t nbytes);

// Read from BO at index
bool fpga_bo_read(int idx, void * dst, size_t nbytes);

// Run the matmul kernel using BO indices. Returns true on success.
bool fpga_run_matmul(int bo_A, int bo_B, int bo_C, int M, int K, int N);

// ================ TASk3================
// map giữa tên tensor và BO index
void fpga_register_tensor_bo(const std::string &name, int bo_idx);

// Lấy BO index từ tên tensor 
int fpga_get_bo_idx_for_name(const std::string &name);
#endif // FPGA_HOST_H
