
#include <stdint.h>
#include <hls_math.h>
#include "KERNAL_FORWARD.h"
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_half.h"


#define TILE_M 16
#define TILE_K 32
#define TILE_N 32


extern "C" {
void KERNAL_FORWARD(
    const float* A,         // [M x K]
    const block_q8_0* B,    // [K x N]
    float* C,               // [M x N]
    int M,
    int K,
    int N
) {

    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=2048
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=2048
    
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1024


    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=M bundle=control
    #pragma HLS INTERFACE s_axilite port=K bundle=control
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

	// ...
	    float A_tile[TILE_M][TILE_K];
	    block_q8_0 B_tile[TILE_K][TILE_N / QK8_0]; // Kích thước là [32][2] với TILE_N=64
	    float C_tile[TILE_M][TILE_N];


	    // Partition A theo chiều M (dim=1)
	    #pragma HLS ARRAY_PARTITION variable=A_tile complete dim=1
	    // Partition B hoàn toàn (dim=0) để truy cập song song 2 block
	    #pragma HLS ARRAY_PARTITION variable=B_tile complete dim=0
	    // Partition C theo chiều N (dim=2)
	    #pragma HLS ARRAY_PARTITION variable=C_tile complete dim=2

//	#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=8 dim=2
	//#pragma HLS ARRAY_PARTITION variable=B_tile cyclic factor=8 dim=2
//	#pragma HLS ARRAY_PARTITION variable=C_tile cyclic factor=8 dim=2




    tile_row_loop: for (int m0 = 0; m0 < M; m0 += TILE_M) {
        tile_col_loop: for (int n0 = 0; n0 < N; n0 += TILE_N) {


            init_C_tile: for (int tm = 0; tm < TILE_M; ++tm) {
                init_C_tile_inner: for (int tn = 0; tn < TILE_N; ++tn) {
                    #pragma HLS PIPELINE
                    C_tile[tm][tn] = 0.0f;
                }
            }

            tile_k_loop: for (int k0 = 0; k0 < K; k0 += TILE_K) {


                load_A_tile: for (int tm = 0; tm < TILE_M; ++tm) {
                    load_A_tile_inner: for (int tk = 0; tk < TILE_K; ++tk) {
                        #pragma HLS PIPELINE
                        if (m0 + tm < M && k0 + tk < K) {
                            A_tile[tm][tk] = A[(m0 + tm)*K + (k0 + tk)];
                        }
                    }
                }

                load_B_tile: for (int tk = 0; tk < TILE_K; ++tk) {
                    load_B_tile_inner: for (int tn_block = 0; tn_block < TILE_N / QK8_0; ++tn_block) {
                        #pragma HLS PIPELINE
                        if (k0 + tk < K && n0 + (tn_block * QK8_0) < N) {
                            int k = k0 + tk;
                            int n = n0 + (tn_block * QK8_0);

                            int block_row_start = k * (N / QK8_0);
                            int block_idx = block_row_start + (n / QK8_0);

                            B_tile[tk][tn_block] = B[block_idx];
                        }
                    }
                }



			compute_loop_M: for (int tm = 0; tm < TILE_M; ++tm) {
				compute_loop_K: for (int tk = 0; tk < TILE_K; ++tk) {

					// Pipeline vòng lặp K
					#pragma HLS PIPELINE II=1

					// Đọc 1 giá trị A, dùng chung cho tất cả N
					float a_val = A_tile[tm][tk];

					compute_loop_N: for (int tn = 0; tn < TILE_N; ++tn) {

						// Unroll vòng lặp N (thực hiện 32 phép toán song song)
						#pragma HLS UNROLL

						// Logic dequantize chính xác
						int tn_block = tn / QK8_0;
						int idx_in_block = tn % QK8_0;

						// Đọc B_tile song song nhờ ARRAY_PARTITION complete
						const block_q8_0 block = B_tile[tk][tn_block];
						const float d = block.d;
						const int8_t b_q = block.qs[idx_in_block];
						float b_val = (float)b_q * d;

						// Tích lũy song song vào 32 thanh ghi C_tile
						C_tile[tm][tn] += a_val * b_val;
					}
				}
			}

            store_C_tile: for (int tm = 0; tm < TILE_M; ++tm) {
                store_C_tile_inner: for (int tn = 0; tn < TILE_N; ++tn) {
                    #pragma HLS PIPELINE
                    if (m0 + tm < M && n0 + tn < N) {
                        C[(m0 + tm)*N + (n0 + tn)] = C_tile[tm][tn];
                    }
                }
            }

        }
    }
}
}
}
