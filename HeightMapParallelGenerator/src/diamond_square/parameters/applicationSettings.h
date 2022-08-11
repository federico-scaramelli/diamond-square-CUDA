#pragma once

// DEBUG
#define PRINT_GRAYSCALE_CUDA	0
#define PRINT_DIAMOND_STEP_CUDA	0
#define PRINT_SQUARE_STEP_CUDA	0

#define PRINT_GRAYSCALE_SEQ		0
#define PRINT_DIAMOND_STEP_SEQ	0
#define PRINT_SQUARE_STEP_SEQ 	0

#define TESTING_SETTINGS		0


// RESULTS
#define SAVE_GRAYSCALE_IMAGE	0
#define SAVE_COLOR_IMAGE		0

#define COPY_RESULT_ON_HOST		1

#define COMPARE_SEQ				1
#define COMPARE_CONSTANT_MEM	1

#define CUDA_EVENTS_TIMING		0


// SETTINGS
#define SQUARE_BLOCK_X_SIZE		8   //(MAX_BLOCK_SIZE / 2 + 1)
#define MAX_BLOCK_SIZE			16	// 16 or 32


#define RANDOM_SINGLE_BATCH		1
#define RANDOM_STEP_BATCHES		0
#define RANDOM_SINGLE_VALUE		1


// CONSTANTS
#define GRAYSCALE_CUDA_PATH				"grayscaleCuda.bmp"
#define COLOR_CUDA_PATH					"colorCuda.bmp"
#define GRAYSCALE_CUDA_CONST_PATH		"grayscaleCudaConst.bmp"
#define COLOR_CUDA_CONST_PATH			"colorCudaConst.bmp"
#define GRAYSCALE_SEQ_PATH				"grayscale.bmp"
#define COLOR_SEQ_PATH					"color.bmp"