// FPGA host implementation. If compiled with -DUSE_FPGA this file
// uses the XRT C++ wrapper to manage device/kernel/BOs. If not,
// it provides simple stub fallbacks so the rest of the project
// can build and run on a development host.

#include "fpga_host.h"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map> 
#include <cstdint>


#ifdef USE_FPGA
// ---- INCLUDE XRT 
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// HP0 memory segment addresses from Vivado design for ZCU102 (full 64-bit values)
// Values derived from the Address Editor in your Vivado export (ZCU102):
//  - HP0_DDR_LOW  : 0x0000_0000_0000_0000 (low DDR window)
//  - HP0_QSPI     : 0x0000_0000_C000_0000 (QSPI mapping, 512M)
//  - HP0_PCIE_LOW : 0x0000_0000_E000_0000 (PCIE low region, 256M)
//  - HP0_DDR_HIGH : 0x0000_0008_0000_0000 (upper DDR mapping, 32G)
//  - CONTROL_REG_BASE: 0x0000_0004_0000_0000 (PS register region used by design)
constexpr uint64_t HP0_DDR_HIGH_BASE = 0x0000000800000000ULL; // 0x0000_0008_0000_0000
constexpr uint64_t HP0_DDR_LOW_BASE  = 0x0000000000000000ULL; // 0x0000_0000_0000_0000
constexpr uint64_t HP0_QSPI_BASE     = 0x00000000C0000000ULL; // 0x0000_0000_C000_0000
constexpr uint64_t HP0_PCIE_LOW_BASE = 0x00000000E0000000ULL; // 0x0000_0000_E000_0000
constexpr uint64_t CONTROL_REG_BASE  = 0x0000000400000000ULL; // 0x0000_0004_0000_0000 (4G region)

// Map bank ID to physical base address for HP0 segments
uint64_t get_bank_base(int bank) {
    switch(bank) {
        // Bank mapping (choose according to how you assigned slave segments in Vivado):
        // 0 -> HP0_DDR_HIGH
        // 1 -> HP0_DDR_LOW
        // 2 -> HP0_QSPI
        // 3 -> HP0_PCIE_LOW
        case 0: return HP0_DDR_HIGH_BASE; // Default high memory
        case 1: return HP0_DDR_LOW_BASE;  // Low memory region
        case 2: return HP0_QSPI_BASE;     // QSPI region
        case 3: return HP0_PCIE_LOW_BASE; // PCIE low region
        default: return HP0_DDR_HIGH_BASE;
    }
}
#endif

static std::mutex g_fpga_mutex;

#ifdef USE_FPGA
static std::unique_ptr<xrt::device> g_device;
static xrt::uuid g_uuid;
static std::unique_ptr<xrt::kernel> g_kernel;
static std::vector<std::unique_ptr<xrt::bo>> g_bos;
static bool g_ready = false;

// --- THÊM CHO TASK 3 ---
// Map để lưu trữ tên tensor -> BO index
static std::unordered_map<std::string, int> g_tensor_to_bo_idx;
// --- KẾT THÚC THÊM ---

// ---  TASK 4 ---
static int g_bo_A_idx = -1; // INDEX CHO BO (A) TOAN CUC 
static int g_bo_C_idx = -1; // INDEX CHO BO (C) TOAN CUC 
// --- END TASK 4 ---

#else
static bool g_ready = false;
static std::vector<int> g_bos_dummy; // indexes only for stubs
static std::unordered_map<std::string, int> g_tensor_to_bo_idx; // THEN CA STUB 
static int g_bo_A_idx = -1; // THEM CA STUB 
static int g_bo_C_idx = -1; // THEM CA STUB 
#endif

bool fpga_host_init(const std::string &xclbin_path, const std::string &kernel_name, std::string &err) {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
#ifdef USE_FPGA
    try {
        g_device = std::make_unique<xrt::device>(0);
        g_uuid = g_device->load_xclbin(xclbin_path);
        g_kernel = std::make_unique<xrt::kernel>(*g_device, g_uuid, kernel_name);
        g_bos.clear();
        g_tensor_to_bo_idx.clear(); // XOA MAP CU 
        g_bo_A_idx = -1; // RESET KHI INIT 
        g_bo_C_idx = -1; // RESET KHI INIT 
        g_ready = true;
        return true;
    } catch (const std::exception &e) {
        std::ostringstream ss;
        ss << "fpga_host_init: exception: " << e.what();
        err = ss.str();
        g_ready = false;
        return false;
    }
#else
    (void)xclbin_path; (void)kernel_name;
    err = "FPGA support disabled at compile time (build with -DUSE_FPGA)";
    g_ready = false;
    return false;
#endif
}

void fpga_host_shutdown() {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
#ifdef USE_FPGA
    g_tensor_to_bo_idx.clear();  // xoa map cu khi init 
    g_bos.clear();
    g_kernel.reset();
    g_device.reset();
    g_bo_A_idx = -1; // Reset khi shutdown
    g_bo_C_idx = -1; // Reset khi shutdown
    g_ready = false;
#else
    g_tensor_to_bo_idx.clear();
    g_bos_dummy.clear();
    g_ready = false;
#endif
}

bool fpga_ready() {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
    return g_ready;
}

int fpga_alloc_bo(size_t bytes, int bank) {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
#ifdef USE_FPGA
    if (!g_device) return -1;
    try {
        // Create BO mapped to correct HP0 segment
        uint64_t base_addr = get_bank_base(bank);
        auto bo = std::make_unique<xrt::bo>(*g_device, bytes, XRT_BO_FLAGS_NORMAL, base_addr);
        g_bos.push_back(std::move(bo));
        return (int)g_bos.size() - 1;
    } catch (...) {
        return -1;
    }
#else
     (void)bytes; (void)bank;
    // stub: return increasing dummy index
    int idx = (int)g_bos_dummy.size();
    g_bos_dummy.push_back((int)bytes);
    return idx;
#endif
}

bool fpga_bo_write(int idx, const void * src, size_t nbytes) {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
#ifdef USE_FPGA
    if ((size_t)idx >= g_bos.size()) return false;
    try {
        auto &bo = *g_bos[idx];
        bo.write(src, nbytes);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return true;
    } catch (...) {
        return false;
    }
#else
    (void)idx; (void)src; (void)nbytes;
    return false;
#endif
}

bool fpga_bo_read(int idx, void * dst, size_t nbytes) {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
#ifdef USE_FPGA
    if ((size_t)idx >= g_bos.size()) return false;
    try {
        auto &bo = *g_bos[idx];
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo.read(dst, nbytes);
        return true;
    } catch (...) {
        return false;
    }
#else
    (void)idx; (void)dst; (void)nbytes;
    return false;
#endif
}

bool fpga_run_matmul(int bo_A, int bo_B, int bo_C, int M, int K, int N) {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
#ifdef USE_FPGA
    if (!g_kernel) return false;
    if ((size_t)bo_A >= g_bos.size() || (size_t)bo_B >= g_bos.size() || (size_t)bo_C >= g_bos.size()) return false;
    try {
        auto run = (*g_kernel)(*g_bos[bo_A], *g_bos[bo_B], *g_bos[bo_C], M, K, N);
        run.wait();
        return true;
    } catch (...) {
        return false;
    }
#else
    (void)bo_A; (void)bo_B; (void)bo_C; (void)M; (void)K; (void)N;
    return false;
#endif
}
// --- ADD TASK 3 ---
void fpga_register_tensor_bo(const std::string &name, int bo_idx) {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
    g_tensor_to_bo_idx[name] = bo_idx;
}

int fpga_get_bo_idx_for_name(const std::string &name) {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
    auto it = g_tensor_to_bo_idx.find(name);
    if (it != g_tensor_to_bo_idx.end()) {
        return it->second;
    }
    return -1; // Không tìm thấy
}
// --- END TASK 3 ---

// --- ADD FOR TASK 4 ---------- 
bool fpga_create_global_buffers(size_t n_ctx, size_t n_ff, std::string &err) {
    std::lock_guard<std::mutex> lk(g_fpga_mutex);
#ifdef USE_FPGA
    if (!g_ready) {
        err = "FPGA not ready";
        return false;
    }
    // TINH TOAN KICH THUOC TOI DA
    // A (src0) LA Q8_0 (1BYTE)
    // C (dst) LA F32 (4 BYTE)
    // KICH THUOC LON NHAT CUA MA TRAN LA (n_ctx) x (n_ff)
    size_t max_rows = n_ctx;
    size_t max_cols = n_ff;
    size_t bytes_A = max_rows * max_cols * sizeof(int8_t); // Kích thước của Q8_0
    size_t bytes_C = max_rows * max_cols * sizeof(float); // Kích thước của F32
    try {
        // CAP PHAT BO CHO A (Activation)
        g_bo_A_idx = fpga_alloc_bo(bytes_A);
        if (g_bo_A_idx < 0) {
            err = "Failed to allocate global BO for A";
            return false;
        }

        // CAP PHAT  BO cho C (Result)
        g_bo_C_idx = fpga_alloc_bo(bytes_C);
        if (g_bo_C_idx < 0) {
            err = "Failed to allocate global BO for C";
            // (Lưu ý: nên có logic thu hồi g_bo_A_idx ở đây, nhưng tạm bỏ qua cho prototype)
            return false;
        }
        
        return true;
    } catch (const std::exception &e) {
        err = e.what();
        return false;
    }
#else
    (void)n_ctx; (void)n_ff;
    err = "FPGA support disabled";
    g_bo_A_idx = 999998; // Dummy index cho stub
    g_bo_C_idx = 999999; // Dummy index cho stub
    return true; // Trả về true cho stub để code chính tiếp tục
#endif
}

int fpga_get_global_bo_A_idx() {
    // Không cần mutex, đây là chỉ số int, đọc_ghi là atomic
    return g_bo_A_idx;
}

int fpga_get_global_bo_C_idx() {
    return g_bo_C_idx;
}
// ------ END TASK 4 -----