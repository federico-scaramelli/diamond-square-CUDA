#include "bmpHandler.h"
#include "settingsList.h"
#include "DiamondSquareSequential.h"
#include "timeMeasure.h"

void run() {
	uint32_t settingIdx = 4;

	try {
		DiamondSquareSequential ds{diamondSquareSettings[settingIdx].size};
		ds.SetRandomScale(diamondSquareSettings[settingIdx].randomScale);
		ds.SetInitialStepSize(diamondSquareSettings[settingIdx].initialStepSize);

		MeasureTimeFn("Algorithm execution: ", 
					  &ds, &DiamondSquareBase::ExecuteDiamondSquare);
		//ds.PrintMap();
		//ds.PrintGrayScaleMap();
		MeasureTimeFn("Grayscale image generation and save file: ",
					  &ds, &DiamondSquareBase::SaveGrayScaleImage,
		              "map.bmp", diamondSquareSettings[settingIdx].imageTileSize);
		MeasureTimeFn("Color image generation and save file: ",
					  &ds, &DiamondSquareBase::SaveColorImage,
		              "mapColor.bmp", diamondSquareSettings[settingIdx].imageTileSize);
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
