#pragma once

// DEBUG
#define PRINT_GRAYSCALE_CUDA	0
#define PRINT_DIAMOND_STEP_CUDA	0
#define PRINT_SQUARE_STEP_CUDA	0

#define PRINT_GRAYSCALE_SEQ		0
#define PRINT_DIAMOND_STEP_SEQ	0
#define PRINT_SQUARE_STEP_SEQ 	0

#define TESTING_SETTINGS		10


// RESULTS
#define SAVE_GRAYSCALE_IMAGE	1
#define SAVE_COLOR_IMAGE		0
#define DELETE_DOUBLE_MAP		0
#define TEST_MAP_FUNC			1

#define COPY_RESULT_ON_HOST		1

#define COMPARE_SEQ				0
#define COMPARE_CONSTANT_MEM	0

#define CURAND_DEVICE			0

#define CUDA_EVENTS_TIMING		0


// SETTINGS
#define SQUARE_BLOCK_X_SIZE		8   // (MAX_BLOCK_SIZE / 2 + 1)
#define MAX_BLOCK_SIZE			16	// 16 or 32
#define BLOCK_SIZE_1D			256	


// CONSTANTS
#define GRAYSCALE_CUDA_PATH				"grayscaleCuda.bmp"
#define COLOR_CUDA_PATH					"colorCuda.bmp"
#define GRAYSCALE_CUDA_CONST_PATH		"grayscaleCudaConst.bmp"
#define COLOR_CUDA_CONST_PATH			"colorCudaConst.bmp"
#define GRAYSCALE_SEQ_PATH				"grayscale.bmp"
#define COLOR_SEQ_PATH					"color.bmp"
