// FPGA host implementation. If compiled with -DUSE_FPGA this file
// uses the XRT C++ wrapper to manage device/kernel/BOs. If not,
// it provides simple stub fallbacks so the rest of the project
// can build and run on a development host.

#include "fpga_host.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <iostream>
#include <cstring>
#include <unordered_map>

// ================= CẤU HÌNH ĐỊA CHỈ (KV260 Bare-metal) =================
constexpr uint64_t KERNEL_CTRL_BASE = 0xA0000000; 
constexpr size_t   KERNEL_CTRL_SIZE = 65536; 
// Địa chỉ vùng nhớ vật lý (cần bootargs mem=1024M để tránh xung đột)
constexpr uint64_t PHY_MEM_BASE     = 0x40000000; 
// =======================================================================

struct MemBuffer {
    uint64_t phy_addr; 
    void* virt_addr; 
    size_t   size;
    bool     valid;
};

// Global state
static int g_mem_fd = -1;
static void* g_ctrl_base_virt = nullptr;
static std::vector<MemBuffer> g_buffers;
static uint64_t g_current_phy_offset = 0;
static std::mutex g_mutex;
static bool g_ready = false;

// Map lưu trữ cặp index (Scale và Data)
struct TensorBOs { int d; int qs; };
static std::unordered_map<std::string, TensorBOs> g_tensor_map;

static int g_bo_A_idx = -1;
static int g_bo_C_idx = -1;

// Helper: Ghi thanh ghi 32-bit
static void reg_write(int offset, uint32_t value) {
    if (g_ctrl_base_virt) {
        volatile uint32_t* ptr = (volatile uint32_t*)((char*)g_ctrl_base_virt + offset);
        *ptr = value;
    }
}

// Helper: Đọc thanh ghi 32-bit
static uint32_t reg_read(int offset) {
    if (g_ctrl_base_virt) {
        volatile uint32_t* ptr = (volatile uint32_t*)((char*)g_ctrl_base_virt + offset);
        return *ptr;
    }
    return 0;
}

bool fpga_host_init(const std::string &xclbin_path, const std::string &kernel_name, std::string &err) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    (void)xclbin_path; (void)kernel_name;
    
    // 1. Mở /dev/mem
    if ((g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) {
        err = "Cannot open /dev/mem. Are you root (sudo)?";
        return false;
    }

    // 2. Map thanh ghi điều khiển
    void* map_base = mmap(0, KERNEL_CTRL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, g_mem_fd, KERNEL_CTRL_BASE);
    if (map_base == MAP_FAILED) {
        err = "mmap control register failed";
        close(g_mem_fd);
        return false;
    }
    g_ctrl_base_virt = map_base;

    // Reset state
    g_buffers.clear();
    g_tensor_map.clear();
    g_current_phy_offset = 0;
    
    // Kiểm tra trạng thái
    uint32_t status = reg_read(0x00);
    std::cout << "[FPGA] Init: Control Reg (0x00) = 0x" << std::hex << status << std::dec << std::endl;

    g_ready = true;
    return true;
#else
    err = "FPGA disabled";
    return false;
#endif
}

void fpga_host_shutdown() {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    for (auto &b : g_buffers) {
        if (b.valid && b.virt_addr) {
            munmap(b.virt_addr, b.size);
        }
    }
    g_buffers.clear();

    if (g_ctrl_base_virt) {
        munmap(g_ctrl_base_virt, KERNEL_CTRL_SIZE);
        g_ctrl_base_virt = nullptr;
    }
    if (g_mem_fd != -1) {
        close(g_mem_fd);
        g_mem_fd = -1;
    }
    g_ready = false;
#endif
}

bool fpga_ready() {
    return g_ready;
}

int fpga_alloc_bo(size_t bytes, int bank) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    (void)bank;
    if (!g_ready) return -1;

    // Cấp phát địa chỉ vật lý tuyến tính
    uint64_t phy_addr = PHY_MEM_BASE + g_current_phy_offset;
    
    // Map vùng nhớ
    void* virt_addr = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g_mem_fd, phy_addr);
    if (virt_addr == MAP_FAILED) {
        std::cerr << "[FPGA] Failed to mmap buffer at phy: 0x" << std::hex << phy_addr << std::dec << std::endl;
        return -1;
    }

    MemBuffer buf;
    buf.phy_addr = phy_addr;
    buf.virt_addr = virt_addr;
    buf.size = bytes;
    buf.valid = true;

    g_buffers.push_back(buf);
    g_current_phy_offset += bytes;
    
    // Align 4KB
    if (g_current_phy_offset % 4096 != 0) {
        g_current_phy_offset += (4096 - (g_current_phy_offset % 4096));
    }

    return (int)g_buffers.size() - 1;
#else
    return -1;
#endif
}

bool fpga_bo_write(int idx, const void * src, size_t nbytes) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if (idx < 0 || idx >= (int)g_buffers.size()) return false;
    MemBuffer &buf = g_buffers[idx];
    if (nbytes > buf.size) nbytes = buf.size;
    std::memcpy(buf.virt_addr, src, nbytes);
    return true;
#else
    return false;
#endif
}

bool fpga_bo_read(int idx, void * dst, size_t nbytes) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if (idx < 0 || idx >= (int)g_buffers.size()) return false;
    MemBuffer &buf = g_buffers[idx];
    if (nbytes > buf.size) nbytes = buf.size;
    std::memcpy(dst, buf.virt_addr, nbytes);
    return true;
#else
    return false;
#endif
}

// Chạy Kernel (Cập nhật offset theo thiết kế mới nhất của bạn)
bool fpga_run_matmul(int bo_A, int bo_B_d, int bo_B_qs, int bo_C, int M, int K, int N) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if (!g_ready) return false;
    
    uint64_t addr_A    = g_buffers[bo_A].phy_addr;
    uint64_t addr_B_d  = g_buffers[bo_B_d].phy_addr;
    uint64_t addr_B_qs = g_buffers[bo_B_qs].phy_addr;
    uint64_t addr_C    = g_buffers[bo_C].phy_addr;

    // A -> 0x10
    reg_write(0x10, (uint32_t)(addr_A & 0xFFFFFFFF));
    reg_write(0x14, (uint32_t)(addr_A >> 32));

    // B_d -> 0x1c
    reg_write(0x1c, (uint32_t)(addr_B_d & 0xFFFFFFFF));
    reg_write(0x20, (uint32_t)(addr_B_d >> 32));

    // B_qs -> 0x28
    reg_write(0x28, (uint32_t)(addr_B_qs & 0xFFFFFFFF));
    reg_write(0x2c, (uint32_t)(addr_B_qs >> 32));

    // C -> 0x34
    reg_write(0x34, (uint32_t)(addr_C & 0xFFFFFFFF));
    reg_write(0x38, (uint32_t)(addr_C >> 32));

    // M -> 0x40
    reg_write(0x40, (uint32_t)M);

    // K -> 0x48
    reg_write(0x48, (uint32_t)K);

    // N -> 0x50
    reg_write(0x50, (uint32_t)N);

    // Start
    reg_write(0x00, 1);

    // Poll
    int timeout = 10000000;
    while (timeout-- > 0) {
        if ((reg_read(0x00) & 0x6)) return true; // Done or Idle
    }
    std::cerr << "[FPGA] Timeout!" << std::endl;
    return false;
#else
    return false;
#endif
}

void fpga_register_tensor_bo(const std::string &name, int bo_d_idx, int bo_qs_idx) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_tensor_map[name] = {bo_d_idx, bo_qs_idx};
}

BO_Pair fpga_get_bo_idx_for_name(const std::string &name) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_tensor_map.find(name);
    if (it != g_tensor_map.end()) {
        return {it->second.d, it->second.qs};
    }
    return {-1, -1};
}

bool fpga_create_global_buffers(size_t n_ctx, size_t n_ff, std::string &err) {
    // A và C đều là float (4 bytes)
    size_t bytes_A = n_ctx * n_ff * 4;
    size_t bytes_C = n_ctx * n_ff * 4;
    
    g_bo_A_idx = fpga_alloc_bo(bytes_A);
    g_bo_C_idx = fpga_alloc_bo(bytes_C);
    
    if (g_bo_A_idx < 0 || g_bo_C_idx < 0) {
        err = "Failed to alloc global buffers (Check PHY_MEM_BASE)";
        return false;
    }
    return true;
}

int fpga_get_global_bo_A_idx() { return g_bo_A_idx; }
int fpga_get_global_bo_C_idx() { return g_bo_C_idx; }

// Helper để ghi trực tiếp (Tránh OOM)
void* fpga_get_virt_addr(int idx) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (idx < 0 || idx >= (int)g_buffers.size()) return nullptr;
    return g_buffers[idx].virt_addr;
}
/*
#include "fpga_host.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include <sstream>

// ================= CẤU HÌNH ĐỊA CHỈ  =================
// Địa chỉ điều khiển (S_AXI_CONTROL)
// Dựa trên Vivado Address Editor: 0xA0000000
constexpr uint64_t KERNEL_CTRL_BASE = 0xA0000000; 

// Kích thước vùng điều khiển
// Dựa trên Vivado: Range = 64K
constexpr size_t   KERNEL_CTRL_SIZE = 65536; 

// Địa chỉ RAM vật lý
// Vẫn dùng vùng 1GB (0x40000000) để an toàn cho KV260
constexpr uint64_t PHY_MEM_BASE     = 0x40000000; 
// =================================================================================


struct MemBuffer {
    uint64_t phy_addr; 
    void* virt_addr; 
    size_t   size;
    bool     valid;
};

static int g_mem_fd = -1;
static void* g_ctrl_base_virt = nullptr;
static std::vector<MemBuffer> g_buffers;
static uint64_t g_current_phy_offset = 0;
static std::mutex g_mutex;
static bool g_ready = false;

//  Map lưu trữ cặp index thay vì 1 int
struct TensorBOs { int d; int qs; };
static std::unordered_map<std::string, TensorBOs> g_tensor_map;

static int g_bo_A_idx = -1;
static int g_bo_C_idx = -1;

static void reg_write(int offset, uint32_t value) {
    if (g_ctrl_base_virt) {
        volatile uint32_t* ptr = (volatile uint32_t*)((char*)g_ctrl_base_virt + offset);
        *ptr = value;
    }
}

static uint32_t reg_read(int offset) {
    if (g_ctrl_base_virt) {
        volatile uint32_t* ptr = (volatile uint32_t*)((char*)g_ctrl_base_virt + offset);
        return *ptr;
    }
    return 0;
}


bool fpga_host_init(const std::string &xclbin_path, const std::string &kernel_name, std::string &err) {
    
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    (void)xclbin_path; (void)kernel_name;
    if ((g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) { err = "Open /dev/mem failed"; return false; }
    void* map = mmap(0, KERNEL_CTRL_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, KERNEL_CTRL_BASE);
    if (map == MAP_FAILED) { close(g_mem_fd); return false; }
    g_ctrl_base_virt = map;
    g_buffers.clear(); g_tensor_map.clear(); g_current_phy_offset = 0;
    g_ready = true;
    return true;
#else
    return false;
#endif
}

void fpga_host_shutdown() {
    
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    for(auto& b: g_buffers) if(b.valid) munmap(b.virt_addr, b.size);
    g_buffers.clear();
    if(g_ctrl_base_virt) munmap(g_ctrl_base_virt, KERNEL_CTRL_SIZE);
    if(g_mem_fd != -1) close(g_mem_fd);
    g_ready = false;
#endif
}

bool fpga_ready() { return g_ready; }

int fpga_alloc_bo(size_t bytes, int bank) {
    
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    (void)bank;
    if(!g_ready) return -1;
    uint64_t phy = PHY_MEM_BASE + g_current_phy_offset;
    void* virt = mmap(0, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, g_mem_fd, phy);
    if(virt == MAP_FAILED) return -1;
    g_buffers.push_back({phy, virt, bytes, true});
    g_current_phy_offset += bytes;
    if(g_current_phy_offset % 4096) g_current_phy_offset += (4096 - (g_current_phy_offset%4096));
    return g_buffers.size() - 1;
#else
    return -1;
#endif
}

bool fpga_bo_write(int idx, const void* src, size_t n) {
   
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if(idx < 0 || idx >= (int)g_buffers.size()) return false;
    if(n > g_buffers[idx].size) n = g_buffers[idx].size;
    memcpy(g_buffers[idx].virt_addr, src, n);
    return true;
#else
    return false;
#endif
}

bool fpga_bo_read(int idx, void* dst, size_t n) {
   
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if(idx < 0 || idx >= (int)g_buffers.size()) return false;
    if(n > g_buffers[idx].size) n = g_buffers[idx].size;
    memcpy(dst, g_buffers[idx].virt_addr, n);
    return true;
#else
    return false;
#endif
}

// 
bool fpga_run_matmul(int bo_A, int bo_B_d, int bo_B_qs, int bo_C, int M, int K, int N) {
    std::lock_guard<std::mutex> lk(g_mutex);
#ifdef USE_FPGA
    if (!g_ready) return false;
    
    // Lấy địa chỉ vật lý
    uint64_t addr_A    = g_buffers[bo_A].phy_addr;
    uint64_t addr_B_d  = g_buffers[bo_B_d].phy_addr;
    uint64_t addr_B_qs = g_buffers[bo_B_qs].phy_addr;
    uint64_t addr_C    = g_buffers[bo_C].phy_addr;

    // A -> 0x10
    reg_write(0x10, (uint32_t)(addr_A & 0xFFFFFFFF));
    reg_write(0x14, (uint32_t)(addr_A >> 32));

    // B_d -> 0x1c
    reg_write(0x1c, (uint32_t)(addr_B_d & 0xFFFFFFFF));
    reg_write(0x20, (uint32_t)(addr_B_d >> 32));

    // B_qs -> 0x28
    reg_write(0x28, (uint32_t)(addr_B_qs & 0xFFFFFFFF));
    reg_write(0x2c, (uint32_t)(addr_B_qs >> 32));

    // C -> 0x34
    reg_write(0x34, (uint32_t)(addr_C & 0xFFFFFFFF));
    reg_write(0x38, (uint32_t)(addr_C >> 32));

    // M, K, N
    reg_write(0x40, (uint32_t)M);
    reg_write(0x48, (uint32_t)K);
    reg_write(0x50, (uint32_t)N);

    // Start
    reg_write(0x00, 1);

    // Poll
    int timeout = 10000000;
    while (timeout-- > 0) {
        if ((reg_read(0x00) & 0x6)) return true; // Done(2) or Idle(4)
    }
    std::cerr << "[FPGA] Timeout!" << std::endl;
    return false;
#else
    return false;
#endif
}

// [THAY ĐỔI] Register nhận 2 BO
void fpga_register_tensor_bo(const std::string &name, int bo_d_idx, int bo_qs_idx) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_tensor_map[name] = {bo_d_idx, bo_qs_idx};
}

// [THAY ĐỔI] Trả về 2 BO
BO_Pair fpga_get_bo_idx_for_name(const std::string &name) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_tensor_map.find(name);
    if (it != g_tensor_map.end()) {
        return {it->second.d, it->second.qs};
    }
    return {-1, -1};
}

// [THAY ĐỔI] Task 4: Tính size A theo Float (4 bytes)
bool fpga_create_global_buffers(size_t n_ctx, size_t n_ff, std::string &err) {
    // A bây giờ là float* => 4 bytes
    size_t bytes_A = n_ctx * n_ff * 4; 
    // C là float* => 4 bytes
    size_t bytes_C = n_ctx * n_ff * 4;
    
    g_bo_A_idx = fpga_alloc_bo(bytes_A);
    g_bo_C_idx = fpga_alloc_bo(bytes_C);
    
    if (g_bo_A_idx < 0 || g_bo_C_idx < 0) {
        err = "Failed to alloc global buffers";
        return false;
    }
    return true;
}

int fpga_get_global_bo_A_idx() { return g_bo_A_idx; }
int fpga_get_global_bo_C_idx() { return g_bo_C_idx; }
void* fpga_get_virt_addr(int idx) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (idx < 0 || idx >= (int)g_buffers.size()) return nullptr;
    return g_buffers[idx].virt_addr;
}
void* fpga_get_virt_addr(int idx) {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (idx < 0 || idx >= (int)g_buffers.size()) return nullptr;
    return g_buffers[idx].virt_addr;
}
/*
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
constexpr uint64_t CONTROL_REG_BASE  = 0x00000000A0000000ULL; // 0x0000_0000_A000_0000 (4G region)

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
      //  case 2: return HP0_QSPI_BASE;     // QSPI region
       // case 3: return HP0_PCIE_LOW_BASE; // PCIE low region
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
*/ 