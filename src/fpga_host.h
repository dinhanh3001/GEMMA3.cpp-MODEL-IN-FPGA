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

// [THAY ĐỔI] Hàm chạy kernel nhận 2 chỉ số cho B: bo_B_d (Scale) và bo_B_qs (Data)
bool fpga_run_matmul(int bo_A, int bo_B_d, int bo_B_qs, int bo_C, int M, int K, int N);

// [THAY ĐỔI] Đăng ký 2 chỉ số BO cho một tensor
void fpga_register_tensor_bo(const std::string &name, int bo_d_idx, int bo_qs_idx);

// [THAY ĐỔI] Lấy struct chứa cặp chỉ số (dùng để trả về từ map)
struct BO_Pair { int d; int qs; };
BO_Pair fpga_get_bo_idx_for_name(const std::string &name);


// =============== TASK 4============== 
// CAP PHAT BOS TOAN CUC CHO ACTIVATIONS (A) VA RESULTS (C) 
// DUA TREN KICH THUOC TOI DA CUA MODEL ( LAY TU HPARAMS )
bool fpga_create_global_buffers(size_t n_ctx, size_t n_ff, std::string &err);

// LAY INDEX CUA BO (A) TOAN CUC 
int fpga_get_global_bo_A_idx();

// LAY INDEX CUA BO (C) TOAN CU 
int fpga_get_global_bo_C_idx();
// --- KET THUC TASK 4 -------- 

#endif // FPGA_HOST_H
