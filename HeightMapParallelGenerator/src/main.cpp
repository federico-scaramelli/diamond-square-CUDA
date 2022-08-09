#include <bitset>

#include "./image/bmpHandler.h"
#include "./diamond_square/parameters/settingsList.h"
#include "./diamond_square/sequential/diamondSquareSequential.h"
#include "./diamond_square/cuda/diamondSquareParallel.h"
#include "./utils/timeMeasure.h"

DiamondSquareSettings setting = Size16385_Step4096_Rnd30;
uint32_t testSetting = 10;

void runSequential()
{
	try {
		/*DiamondSquareSequential ds{diamondSquareSettings[setting].size};
		ds.SetRandomScale(diamondSquareSettings[setting].randomScale);
		ds.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);*/
		DiamondSquareSequential ds{testingDiamondSquareSettings[testSetting].size};
		ds.SetRandomScale(30.f);
		ds.SetInitialStepSize(testingDiamondSquareSettings[testSetting].initialStepSize);

		MeasureTimeFn("Sequential algorithm execution: ",
		              &ds, &DiamondSquareBase::ExecuteDiamondSquare);
		MeasureTimeFn("Color image generation and save file: ",
		              &ds, &DiamondSquareBase::SaveColorImage,
		              "mapColorSeq.bmp", diamondSquareSettings[setting].imageTileSize);
		//ds.PrintGrayScaleMap();
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
		ds.SetRandomScale(100);
		ds.SetInitialStepSize(diamondSquareSettings[setting].initialStepSize);
		ds.SetInitialStepSize(32);
		/*DiamondSquareParallel ds{testingDiamondSquareSettings[testSetting].size};
		ds.SetRandomScale(30.f);
		ds.SetInitialStepSize(testingDiamondSquareSettings[testSetting].initialStepSize);*/

		MeasureTimeFn("Parallel algorithm execution: ",
		              &ds, &DiamondSquareBase::ExecuteDiamondSquare);

		//ds.PrintGrayScaleMap();
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
	MeasureTimeFn("Total parallel execution time: ", runParallel);
	std::cout << std::endl;
	//MeasureTimeFn("Total sequential execution time: ", runSequential);
	return (0);
}
