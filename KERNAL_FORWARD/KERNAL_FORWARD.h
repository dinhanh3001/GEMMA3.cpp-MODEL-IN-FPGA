#pragma once

#include <stdint.h>
#include <hls_math.h>

#define QK8_0 32

typedef half float16_t;

struct block_q8_0 {
    float16_t d;
    int8_t qs[QK8_0];
};

extern "C" {
void KERNAL_FORWARD(
	    const float* A,       // [M x K]
	    const half* B_d,     // [K x N/32] - Scale
	    const int8_t* B_qs,   // [K x N]    - Quantized int8
	    float* C,             // [M x N]
	    int M,
	    int K,
	    int N
	);
} // extern "C"
