#include <bitset>

#include "./image/bmpHandler.h"
#include "./diamond_square/parameters/settingsList.h"
#include "./diamond_square/sequential/diamondSquareSequential.h"
#include "./diamond_square/cuda/diamondSquareParallel.h"
#include "./utils/timeMeasure.h"

DiamondSquareSettings setting = Size33_Step32_Rnd1;

void runSequential()
{

	try {
		DiamondSquareSequential ds{diamondSquareSettings[setting].size};
		ds.SetRandomScale(diamondSquareSettings[setting].randomScale);
		ds.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);

		MeasureTimeFn("Sequential algorithm execution: ",
		              &ds, &DiamondSquareBase::ExecuteDiamondSquare);
		//ds.PrintMap();
		//ds.PrintGrayScaleMap();
		/*MeasureTimeFn("Grayscale image generation and save file: ",
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

void runParallel()
{
	try {
		DiamondSquareParallel ds{diamondSquareSettings[setting].size};
		ds.SetRandomScale(diamondSquareSettings[setting].randomScale);
		ds.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);

		MeasureTimeFn("Parallel algorithm execution: ",
		              &ds, &DiamondSquareBase::ExecuteDiamondSquare);

		ds.PrintMap();
		MeasureTimeFn("Color image generation and save file: ",
					  &ds, &DiamondSquareBase::SaveColorImage,
		              "mapColor.bmp", diamondSquareSettings[setting].imageTileSize);

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
	runParallel();
	runSequential();
	/*MeasureTimeFn("Total parallel execution time: ", runParallel);
	MeasureTimeFn("Total sequential execution time: ", runSequential);*/
	return (0);
}
