#include "bmpHandler.h"
#include "DiamondSquareSequential.h"

int main(int argc, char** argv) {

	const uint32_t size = 513;

	try {
		DiamondSquareSequential ds{size};
		ds.ExecuteDiamondSquare();
		//ds.PrintMap();
		//ds.PrintGrayScaleMap();
		ds.SaveMapOnImage("map.bmp", 1);
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return(0);
	}

	return(0);
}