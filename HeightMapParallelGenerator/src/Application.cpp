#include <bitset>
#include <cuda_runtime_api.h>

#include "./image/bmpHandler.h"
#include "./diamond_square/parameters/algorithmSettings.h"
#include "./diamond_square/parameters/applicationSettings.h"
#include "./diamond_square/sequential/diamondSquareSequential.h"
#include "./diamond_square/cuda/diamondSquareParallel.h"
#include "./diamond_square/cuda/DiamondSquareParallel_Constant.h"


DiamondSquareSettings setting = Size16385_Step4096_Rnd30;


#if TESTING_SETTINGS
uint32_t size = testingDiamondSquareSettings[TESTING_SETTINGS].size;
#else
uint32_t size = diamondSquareSettings[setting].size;
#endif


DiamondSquareParallel parDiamSquare{ size };
DiamondSquareSequential seqDiamSquare{ size };
DiamondSquareParallel_Constant parDiamSquareConstMem{ size };


double sequentialTime;
double parallelTime;
double parallelTime_const;

double seqTimeGrayscale;
double parTimeGrayscale;
double seqTimeCustomRangeMap;
double parTimeCustomRangeMap;

// SEQUENTIAL
void runSequential ()
{
#if !TESTING_SETTINGS
	seqDiamSquare.SetRandomScale (diamondSquareSettings[setting].randomScale);
	//ds.SetRandomScale(100);
	seqDiamSquare.SetInitialStepSize (diamondSquareSettings[setting].initialStepSize);
	//ds.SetInitialStepSize(32);
#else
	seqDiamSquare.SetRandomScale(testingDiamondSquareSettings[TESTING_SETTINGS].randomScale);
	seqDiamSquare.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

	MeasureTimeFn (&sequentialTime, "Sequential algorithm execution terminated in ",
	               &seqDiamSquare, &DiamondSquareBase::ExecuteDiamondSquare);

#if PRINT_GRAYSCALE_SEQ
	ds.PrintGrayScaleMap();
#endif

	MeasureTimeFn (&seqTimeCustomRangeMap, "Custom range map generated in ",
				   &seqDiamSquare, &DiamondSquareBase::MapValuesToIntRange,
					-1000, 2000);
	MeasureTimeFn (&seqTimeGrayscale, "Grayscale map generated in ",
				   &seqDiamSquare, &DiamondSquareBase::MapValuesToGrayScale);

#if SAVE_GRAYSCALE_IMAGE
	std::cout << "\n - GRAYSCALE IMAGE - " << std::endl;
	MeasureTimeFn(nullptr, "Grayscale image generated and saved on file in ",
				  &seqDiamSquare, &DiamondSquareBase::SaveGrayScaleImage,
	              GRAYSCALE_SEQ_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	std::cout << "\n - COLOR IMAGE - " << std::endl;
	MeasureTimeFn (nullptr, "Color image generated and saved on file in ",
	               &seqDiamSquare, &DiamondSquareBase::SaveColorImage,
	               COLOR_SEQ_PATH, 1);
#endif
}


// PARALLEL
void runParallel ()
{
#if !TESTING_SETTINGS
	parDiamSquare.SetRandomScale (diamondSquareSettings[setting].randomScale);
	//ds.SetRandomScale(100);
	parDiamSquare.SetInitialStepSize (diamondSquareSettings[setting].initialStepSize);
	//ds.SetInitialStepSize(32);
#else
	parDiamSquare.SetRandomScale(testingDiamondSquareSettings[TESTING_SETTINGS].randomScale);
	parDiamSquare.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

	MeasureTimeFn (&parallelTime, "Parallel algorithm execution terminated in ",
	               &parDiamSquare, &DiamondSquareBase::ExecuteDiamondSquare);

#if PRINT_GRAYSCALE_CUDA
	parDiamSquare.PrintGrayScaleMap();
#endif

	MeasureTimeFn (&parTimeCustomRangeMap, "Custom range map generated in ",
				   &parDiamSquare, &DiamondSquareParallel::MapValuesToIntRange,
					-1000, 2000);
	MeasureTimeFn (&parTimeGrayscale, "Grayscale map generated in ",
				   &parDiamSquare, &DiamondSquareBase::MapValuesToGrayScale);
	

#if SAVE_GRAYSCALE_IMAGE
	std::cout << "\n - GRAYSCALE IMAGE - " << std::endl;
	MeasureTimeFn(nullptr, "Grayscale image generated and saved on file in ",
				  &parDiamSquare, &DiamondSquareBase::SaveGrayScaleImage,
	              GRAYSCALE_CUDA_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	std::cout << "\n - COLOR IMAGE - " << std::endl;
	MeasureTimeFn (nullptr, "Color image generated and saved on file in ",
	               &parDiamSquare, &DiamondSquareBase::SaveColorImage,
	               COLOR_CUDA_PATH, 1);
#endif
}


// PARALLEL CONSTANT MEMORY
void runParallelConstantMem ()
{
#if !TESTING_SETTINGS
	parDiamSquareConstMem.SetRandomScale (diamondSquareSettings[setting].randomScale);
	//ds.SetRandomScale(100);
	parDiamSquareConstMem.SetInitialStepSize (diamondSquareSettings[setting].initialStepSize);
	//ds.SetInitialStepSize(32);
#else
	parDiamSquareConstMem.SetRandomScale(30.f);
	parDiamSquareConstMem.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

	MeasureTimeFn (&parallelTime_const, "Parallel algorithm execution terminated in ",
	               &parDiamSquareConstMem, &DiamondSquareBase::ExecuteDiamondSquare);

#if PRINT_GRAYSCALE_CUDA
	parDiamSquare.PrintGrayScaleMap();
#endif

	MeasureTimeFn (nullptr, "Grayscale map generated in ",
			   &parDiamSquareConstMem, &DiamondSquareBase::MapValuesToGrayScale);

#if SAVE_GRAYSCALE_IMAGE
	std::cout << "\n - GRAYSCALE IMAGE - " << std::endl;
	MeasureTimeFn(nullptr, "Grayscale generated and saved on file in ",
					  &parDiamSquareConstMem, &DiamondSquareBase::SaveGrayScaleImage,
		              GRAYSCALE_CUDA_CONST_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	std::cout << "\n - COLOR IMAGE - " << std::endl;
	MeasureTimeFn (nullptr, "Color generated and saved on file in ",
	               &parDiamSquareConstMem, &DiamondSquareBase::SaveColorImage,
	               COLOR_CUDA_CONST_PATH, 1);
#endif
}


int main (int argc, char** argv)
{
	cudaFree(0);

	try {
		runParallel();
		std::cout << std::endl << std::endl << std::endl << std::endl;

#if COMPARE_CONSTANT_MEM
		runParallelConstantMem();
#endif

#if COMPARE_SEQ
		runSequential();
#endif

#if COMPARE_SEQ || COMPARE_CONSTANT_MEM
		std::cout << std::endl << std::endl << std::endl << std::endl;

		std::cout << " === SPEED-UP RESULTS === " << std::endl;
#if COMPARE_CONSTANT_MEM
		CompareTime("Constant memory usage speed-up is ", parDiamSquare.GetExecutionTime(), parDiamSquareConstMem.GetExecutionTime());
#endif
#if COMPARE_SEQ
		CompareTime ("Diamond Square algorithm parallel speed-up is ", seqDiamSquare.GetExecutionTime(), parDiamSquare.GetExecutionTime());
		CompareTime ("Initialization and algorithm parallel speed-up is ", &sequentialTime, &parallelTime);
		CompareTime ("Grayscale map conversion parallel speed-up is ", &seqTimeGrayscale, &parTimeGrayscale);
		CompareTime ("Custom range map conversion parallel speed-up is ", &seqTimeCustomRangeMap, &parTimeCustomRangeMap);
#endif
#endif
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return 0;
	}
	std::cout << std::endl << std::endl;

	return (0);
}
