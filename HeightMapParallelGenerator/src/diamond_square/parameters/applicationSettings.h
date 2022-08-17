#pragma once

// DEBUG
#define PRINT_GRAYSCALE_CUDA	0
#define PRINT_DIAMOND_STEP_CUDA	0
#define PRINT_SQUARE_STEP_CUDA	0

#define PRINT_GRAYSCALE_SEQ		0
#define PRINT_DIAMOND_STEP_SEQ	0
#define PRINT_SQUARE_STEP_SEQ 	0

#define TESTING_SETTINGS		9


// RESULTS
#define SAVE_GRAYSCALE_IMAGE	0
#define SAVE_COLOR_IMAGE		0

#define DELETE_FLOAT_MAP		1	//Delete the float map when the grayscale is generated

#define COPY_RESULT_ON_HOST		1

#define COMPARE_SEQ				0
#define COMPARE_CONSTANT_MEM	0

#define RAND_DEVICE_API			0

#define EVENTS_TIMING			1


// SETTINGS
#define SQUARE_BLOCK_X_SIZE		8   
#define MAX_BLOCK_SIZE			16	// 16 or 32
#define BLOCK_SIZE_1D			256	
#define MAX_GRID_SIZE_1D		128	


// CONSTANTS
#define GRAYSCALE_CUDA_PATH				"Grayscale_Cuda.bmp"
#define COLOR_CUDA_PATH					"Color_Cuda.bmp"
#define GRAYSCALE_CUDA_CONST_PATH		"Grayscale_CudaConst.bmp"
#define COLOR_CUDA_CONST_PATH			"Color_CudaConst.bmp"
#define GRAYSCALE_SEQ_PATH				"Grayscale.bmp"
#define COLOR_SEQ_PATH					"Color.bmp"
