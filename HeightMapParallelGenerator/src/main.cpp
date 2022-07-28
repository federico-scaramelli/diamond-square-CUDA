#include "bmpHandler.h"
#include "DiamondSquareSequential.h"

int main(int argc, char** argv) {

	const uint32_t size = 65;

	if (!DiamondSquareBase::CheckSizeAdequate(size))
		return(0);

	DiamondSquareSequential ds{size};

	ds.ExecuteDiamondSquare(64);
	ds.PrintMap();
	ds.SaveMapOnImage("map.bmp", 1);

	return(0);
}

void testImage() {
	Color color(std::string("0xFF0000"));
	Color color2("0x0000FF");
	Color colorW(true);
	Color colorB(false);

	BMP image(800, 600);
	image.FillRegion(50, 20, 100, 200, color, 255);
	image.FillRegion(200, 200, 100, 200, color2, 255);
	image.FillRegion(300, 400, 100, 20, colorW, 255);
	image.FillRegion(500, 500, 200, 20, colorB, 255);

	image.Write("prova.bmp");

	BMP image2("prova.bmp");
	image2.FillRegion(600, 100, 200, 300, 255, 0, 0, 255);
	image2.Write("prova.bmp");
}