#include "./image/bmpHandler.h"
#include "./diamond_square/parameters/settingsList.h"
#include "./diamond_square/sequential/diamondSquareSequential.h"
#include "./utils/timeMeasure.h"

void run() {
	DiamondSquareSettings setting = Size4097_Step1024_Rnd15;

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

int main(int argc, char** argv) {

	MeasureTimeFn("Total execution time: ", run);
	return (0);
}
