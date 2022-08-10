#include <bitset>

#include "./image/bmpHandler.h"
#include "./diamond_square/parameters/algorithmSettings.h"
#include "./diamond_square/parameters/applicationSettings.h"
#include "./diamond_square/sequential/diamondSquareSequential.h"
#include "./diamond_square/cuda/diamondSquareParallel.h"

//DiamondSquareSettings setting = Size16385_Step4096_Rnd30;
DiamondSquareSettings setting = Size513_Step256_Rnd5;
DiamondSquareParallel parDiamSquare{diamondSquareSettings[setting].size};
DiamondSquareSequential seqDiamSquare{diamondSquareSettings[setting].size};

double sequentialTime;
double parallelTime;

void runSequential()
{
	try {
#if !TESTING_SETTINGS 
		seqDiamSquare.SetRandomScale(diamondSquareSettings[setting].randomScale);
		//ds.SetRandomScale(100);
		seqDiamSquare.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);
		//ds.SetInitialStepSize(32);
#else
		DiamondSquareSequential ds{testingDiamondSquareSettings[TESTING_SETTINGS].size};
		ds.SetRandomScale(30.f);
		ds.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

		MeasureTimeFn(&sequentialTime, "\nSequential algorithm execution terminated in ",
		              &seqDiamSquare, &DiamondSquareBase::ExecuteDiamondSquare);

#if PRINT_GRAYSCALE_SEQ
		ds.PrintGrayScaleMap();
#endif

#if SAVE_GRAYSCALE_IMAGE
		MeasureTimeFn("Grayscale image generation and save file: ",
					  &ds, &DiamondSquareBase::SaveGrayScaleImage,
		              GRAYSCALE_SEQ_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
		MeasureTimeFn("Color image generation and save file: ",
					  &ds, &DiamondSquareBase::SaveColorImage,
		              COLOR_SEQ_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return;
	}
}

void runParallel()
{
	try {

#if !TESTING_SETTINGS
		parDiamSquare.SetRandomScale(diamondSquareSettings[setting].randomScale);
		//ds.SetRandomScale(100);
		parDiamSquare.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);
		//ds.SetInitialStepSize(32);
#elif DEBUG_MODE
		DiamondSquareParallel ds{testingDiamondSquareSettings[TESTING_SETTINGS].size};
		ds.SetRandomScale(30.f);
		ds.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

		MeasureTimeFn(&parallelTime, "Parallel algorithm execution terminated in ",
		              &parDiamSquare, &DiamondSquareBase::ExecuteDiamondSquare);

#if PRINT_GRAYSCALE_CUDA
		ds.PrintGrayScaleMap();
#endif

#if SAVE_GRAYSCALE_IMAGE
		MeasureTimeFn("Grayscale generated and saved on file in ",
					  &ds, &DiamondSquareBase::SaveGrayScaleImage,
		              GRAYSCALE_CUDA_PATH, diamondSquareSettings[setting].imageTileSize);
#endif
#if SAVE_COLOR_IMAGE
		MeasureTimeFn("Color generated and saved on file in ",
					  &ds, &DiamondSquareBase::SaveColorImage,
		              COLOR_CUDA_PATH, diamondSquareSettings[setting].imageTileSize);
#endif

	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return;
	}
}

int main(int argc, char** argv)
{
	runParallel();

#if COMPARE
	std::cout << std::endl << std::endl << std::endl << std::endl;
	runSequential();
	std::cout << std::endl << std::endl << std::endl << std::endl;
	std::cout << "================SPEED-UP RESULTS================" << std::endl;
	CompareTime ("Initialization and Diamond Square parallel speed-up is ", &sequentialTime, &parallelTime);
	CompareTime ("Diamond Square parallel speed-up is ", seqDiamSquare.GetExecutionTime(), parDiamSquare.GetExecutionTime());
#endif

	std::cout << std::endl << std::endl;

	return (0);
}
