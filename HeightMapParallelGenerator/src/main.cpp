#include <bitset>

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

	MeasureTimeFn (&sequentialTime, "\nSequential algorithm execution terminated in ",
	               &seqDiamSquare, &DiamondSquareBase::ExecuteDiamondSquare);

#if PRINT_GRAYSCALE_SEQ
		ds.PrintGrayScaleMap();
#endif

#if SAVE_GRAYSCALE_IMAGE
		MeasureTimeFn(nullptr, "Grayscale image generation and save file: ",
					  &seqDiamSquare, &DiamondSquareBase::SaveGrayScaleImage,
		              GRAYSCALE_SEQ_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	MeasureTimeFn (nullptr, "Color image generation and save file: ",
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
		parDiamSquare.SetRandomScale(30.f);
		parDiamSquare.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

	MeasureTimeFn (&parallelTime, "Parallel algorithm execution terminated in ",
	               &parDiamSquare, &DiamondSquareBase::ExecuteDiamondSquare);

#if PRINT_GRAYSCALE_CUDA
		parDiamSquare.PrintGrayScaleMap();
#endif

#if SAVE_GRAYSCALE_IMAGE
		MeasureTimeFn(nullptr, "Grayscale generated and saved on file in ",
					  &parDiamSquare, &DiamondSquareBase::SaveGrayScaleImage,
		              GRAYSCALE_CUDA_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	MeasureTimeFn (nullptr, "Color generated and saved on file in ",
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

#if SAVE_GRAYSCALE_IMAGE
		MeasureTimeFn(nullptr, "Grayscale generated and saved on file in ",
					  &parDiamSquareConstMem, &DiamondSquareBase::SaveGrayScaleImage,
		              GRAYSCALE_CUDA_CONST_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
	MeasureTimeFn (nullptr, "Color generated and saved on file in ",
	               &parDiamSquareConstMem, &DiamondSquareBase::SaveColorImage,
	               COLOR_CUDA_CONST_PATH, 1);
#endif
}





int main (int argc, char** argv)
{
	try {
		runParallel();
		std::cout << std::endl << std::endl << std::endl << std::endl;


#if COMPARE_CONSTANT_MEM
		runParallelConstantMem();
		std::cout << std::endl << std::endl;
		CompareTime("Constant memory usage speed-up is ", parDiamSquare.GetExecutionTime(), parDiamSquareConstMem.GetExecutionTime());
		std::cout << std::endl << std::endl << std::endl << std::endl;
#endif


#if COMPARE_SEQ
		runSequential();
		std::cout << std::endl << std::endl << std::endl << std::endl;

		std::cout << "================SPEED-UP RESULTS================" << std::endl;
		CompareTime ("Initialization and Diamond Square parallel speed-up is ", &sequentialTime, &parallelTime);
		CompareTime ("Diamond Square parallel speed-up is ", seqDiamSquare.GetExecutionTime(), parDiamSquare.GetExecutionTime());
#endif
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return 0;
	}
	std::cout << std::endl << std::endl;

	return (0);
}
