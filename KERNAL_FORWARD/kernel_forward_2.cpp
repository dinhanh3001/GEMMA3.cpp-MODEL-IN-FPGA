#include <stdint.h>
#include <hls_math.h>
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_half.h"

#define QK8_0 32

// Định nghĩa lại kiểu dữ liệu để HLS hiểu rõ độ rộng bus
// KV260 bus 128-bit -> tối ưu nhất là đọc data gói gọn trong 128-bit
typedef half float16_t;

// Kích thước Tile cho KV260 (Zynq MPSoC có ít BRAM hơn Alveo, nên giữ tile vừa phải)
#define TILE_M 16
#define TILE_K 64
#define TILE_N 64

extern "C" {

/* 
   Tách block_q8_0 B thành:
   - B_d: Mảng chứa các giá trị scale (float16)
   - B_qs: Mảng chứa các giá trị int8
   Điều này giúp bộ nhớ thẳng hàng (aligned), HLS đọc cực nhanh qua m_axi.
*/
void KERNAL_FORWARD(
    const float* A,       // [M x K]
    const half* B_d,     // [K x N/32] - Scale
    const int8_t* B_qs,   // [K x N]    - Quantized int8
    float* C,             // [M x N]
    int M,
    int K,
    int N
) {
    // Cấu hình giao tiếp AXI Master (Tự động DMA)
    // depth chỉ để mô phỏng C-sim, không ảnh hưởng synthesis
    #pragma HLS INTERFACE m_axi port=A    offset=slave bundle=gmem0 depth=4096 max_read_burst_length=16 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=B_d  offset=slave bundle=gmem1 depth=4096 max_read_burst_length=16 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=B_qs offset=slave bundle=gmem2 depth=4096 max_read_burst_length=16 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=C    offset=slave bundle=gmem3 depth=4096 max_write_burst_length=16 num_write_outstanding=16

    #pragma HLS INTERFACE s_axilite port=M bundle=control
    #pragma HLS INTERFACE s_axilite port=K bundle=control
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Bộ nhớ đệm cục bộ (BRAM/URAM trên KV260)
    float A_tile[TILE_M][TILE_K];
    half  B_d_tile[TILE_K][TILE_N / QK8_0];
    int8_t B_qs_tile[TILE_K][TILE_N];       // Flatten mảng qs để dễ truy cập
    float C_tile[TILE_M][TILE_N];

    // --- FIX SYNTHESIS HANG: Dùng CYCLIC thay vì COMPLETE ---
    #pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=16 dim=2
    // B_d ít truy cập hơn, partition nhỏ cũng được
    #pragma HLS ARRAY_PARTITION variable=B_d_tile cyclic factor=8 dim=2
    // B_qs cần băng thông cao cho tính toán
    #pragma HLS ARRAY_PARTITION variable=B_qs_tile cyclic factor=16 dim=2
    #pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=16 dim=2

    // Main Loops
    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        for (int n0 = 0; n0 < N; n0 += TILE_N) {

            // Xóa C_tile
            init_C: for(int i=0; i<TILE_M; i++)
                for(int j=0; j<TILE_N; j++)
                    #pragma HLS PIPELINE
                    C_tile[i][j] = 0;

            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                // DATAFLOW: Cho phép Load và Compute chạy song song
                #pragma HLS DATAFLOW

                // 1. Load A
                load_A: for (int tm = 0; tm < TILE_M; ++tm) {
                    for (int tk = 0; tk < TILE_K; ++tk) {
                        #pragma HLS PIPELINE II=1
                        if (m0 + tm < M && k0 + tk < K)
                            A_tile[tm][tk] = A[(m0 + tm)*K + (k0 + tk)];
                        else
                            A_tile[tm][tk] = 0.0f;
                    }
                }

                // 2. Load B (Tách ra 2 luồng đọc: Scale và Quants)
                load_B: for (int tk = 0; tk < TILE_K; ++tk) {
                    for (int tn = 0; tn < TILE_N; ++tn) {
                         #pragma HLS PIPELINE II=1
                         if (k0 + tk < K && n0 + tn < N) {
                             // Load Quantized Value
                             B_qs_tile[tk][tn] = B_qs[(k0 + tk)*N + (n0 + tn)];

                             // Load Scale (chỉ load khi hết block 32)
                             if (tn % QK8_0 == 0) {
                                 B_d_tile[tk][tn / QK8_0] = B_d[(k0 + tk)*(N/QK8_0) + (n0 + tn)/QK8_0];
                             }
                         }
                    }
                }

                // 3. Compute (Matrix Multiplication)
                compute: for (int tm = 0; tm < TILE_M; ++tm) {
                    for (int tk = 0; tk < TILE_K; ++tk) {
                        #pragma HLS PIPELINE II=1
                        float a_val = A_tile[tm][tk];

                        for (int tn = 0; tn < TILE_N; ++tn) {
                            // UNROLL tương ứng với factor của Partition (16)
                            #pragma HLS UNROLL factor=16

                            int8_t val_q = B_qs_tile[tk][tn];
                            half val_d = B_d_tile[tk][tn / QK8_0];

                            // Ép kiểu tường minh để tránh lỗi synthesis
                            float decoded_b = (float)val_q * (float)val_d;
                            C_tile[tm][tn] += a_val * decoded_b;
                        }
                    }
                }
            } // end k0

            // 4. Store C
            store_C: for (int tm = 0; tm < TILE_M; ++tm) {
                for (int tn = 0; tn < TILE_N; ++tn) {
                    #pragma HLS PIPELINE II=1
                    if (m0 + tm < M && n0 + tn < N)
                         C[(m0 + tm)*N + (n0 + tn)] = C_tile[tm][tn];
                }
            }

        } // end n0
    } // end m0
}
}
