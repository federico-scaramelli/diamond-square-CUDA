#include <bitset>

#include "./image/bmpHandler.h"
#include "./diamond_square/parameters/algorithmSettings.h"
#include "./diamond_square/parameters/applicationSettings.h"
#include "./diamond_square/sequential/diamondSquareSequential.h"
#include "./diamond_square/cuda/diamondSquareParallel.h"

DiamondSquareSettings setting = Size16385_Step512_Rnd50;

void runSequential()
{
	try {
#if !TESTING_SETTINGS 
		DiamondSquareSequential ds{diamondSquareSettings[setting].size};
		ds.SetRandomScale(diamondSquareSettings[setting].randomScale);
		//ds.SetRandomScale(100);
		ds.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);
		//ds.SetInitialStepSize(32);
#else
		DiamondSquareSequential ds{testingDiamondSquareSettings[TESTING_SETTINGS].size};
		ds.SetRandomScale(30.f);
		ds.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

		MeasureTimeFn("\nSequential algorithm execution terminated in ",
		              &ds, &DiamondSquareBase::ExecuteDiamondSquare);

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
		DiamondSquareParallel ds{diamondSquareSettings[setting].size};
		ds.SetRandomScale(diamondSquareSettings[setting].randomScale);
		//ds.SetRandomScale(100);
		ds.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);
		//ds.SetInitialStepSize(32);
#elif DEBUG_MODE
		DiamondSquareParallel ds{testingDiamondSquareSettings[TESTING_SETTINGS].size};
		ds.SetRandomScale(30.f);
		ds.SetInitialStepSize(testingDiamondSquareSettings[TESTING_SETTINGS].initialStepSize);
#endif

		MeasureTimeFn("\nParallel algorithm execution terminated in ",
		              &ds, &DiamondSquareBase::ExecuteDiamondSquare);
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
	//MeasureTimeFn("\nParallel map generation terminated in ", runParallel);
#if COMPARE
	std::cout << std::endl << std::endl << std::endl << std::endl;
	runSequential();
	//MeasureTimeFn("\nSequential map generation terminated in ", runSequential);
#endif
	return (0);
}
