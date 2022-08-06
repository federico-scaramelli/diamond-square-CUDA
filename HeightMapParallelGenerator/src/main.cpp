#include <bitset>

#include "./image/bmpHandler.h"
#include "./diamond_square/parameters/settingsList.h"
#include "./diamond_square/sequential/diamondSquareSequential.h"
#include "./diamond_square/cuda/diamondSquareParallel.h"
#include "./utils/timeMeasure.h"

void runSequential()
{
	DiamondSquareSettings setting = Size16385_Step4096_Rnd30;

	try {
		DiamondSquareSequential ds{diamondSquareSettings[setting].size};
		ds.SetRandomScale(diamondSquareSettings[setting].randomScale);
		ds.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);

		MeasureTimeFn("Algorithm execution: ",
		              &ds, &DiamondSquareBase::ExecuteDiamondSquare);
		//ds.PrintMap();
		//ds.PrintGrayScaleMap();
		MeasureTimeFn("Grayscale image generation and save file: ",
		              &ds, &DiamondSquareBase::SaveGrayScaleImage,
		              "map.bmp", diamondSquareSettings[setting].imageTileSize);
		MeasureTimeFn("Color image generation and save file: ",
		              &ds, &DiamondSquareBase::SaveColorImage,
		              "mapColor.bmp", diamondSquareSettings[setting].imageTileSize);
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return;
	}
}

void runParallel()
{
	DiamondSquareSettings setting = Size513_Step256_Rnd5;

	try {
		DiamondSquareParallel ds{diamondSquareSettings[setting].size};
		ds.SetRandomScale(diamondSquareSettings[setting].randomScale);
		ds.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);

		ds.InitializeDiamondSquare();

		/*MeasureTimeFn("Algorithm execution: ", 
					  &ds, &DiamondSquareBase::ExecuteDiamondSquare);
		//ds.PrintMap();
		//ds.PrintGrayScaleMap();
		MeasureTimeFn("Grayscale image generation and save file: ",
					  &ds, &DiamondSquareBase::SaveGrayScaleImage,
		              "map.bmp", diamondSquareSettings[setting].imageTileSize);
		MeasureTimeFn("Color image generation and save file: ",
					  &ds, &DiamondSquareBase::SaveColorImage,
		              "mapColor.bmp", diamondSquareSettings[setting].imageTileSize);*/
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return;
	}
}

int main(int argc, char** argv)
{
	//runParallel();
	MeasureTimeFn("Total execution time: ", runParallel);
	return (0);
}
