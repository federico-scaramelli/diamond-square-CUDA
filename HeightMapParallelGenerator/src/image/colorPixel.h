#pragma once

//std headers
#include <cstdint>
#include <iostream>
#include <sstream>

//My headers
#include "../utils/utils.h"

class ColorPixel {
public:

#pragma region Constructors

	//Default constructor
	ColorPixel(const uint8_t B, const uint8_t G, const uint8_t R) {
		this->B = B;
		this->G = G;
		this->R = R;
	}

	//Grayscale color constructor
	ColorPixel(const uint8_t grayValue) {
		this->B = grayValue;
		this->G = grayValue;
		this->R = grayValue;
	}

	//Hex string constructor
	ColorPixel(const char* hexString) {
		if(hexString[0] == '#') {
			hexString++;
		}

		int n;
		std::istringstream(hexString) >> std::hex >> n;

		R = n >> 16 & 0xff;
		//std::cout << (int)R << std::endl;
		G = n >> 8 & 0xff;
		//std::cout << (int)G << std::endl;
		B = n & 0xff;
		//std::cout << (int)B << std::endl;
	}

	//Double constructor with values mapping
	ColorPixel(const float B, const float G, const float R, const float min, const float max) {
		this->B = MapValue(min, max, 0, 255, B);
		this->G = MapValue(min, max, 0, 255, G);
		this->R = MapValue(min, max, 0, 255, R);
	}

	//Black or white constructor
	ColorPixel(const bool white) {
		if (white) {
			this->B = 255;
			this->G = 255;
			this->R = 255;	
		} else {
			this->B = 0;
			this->G = 0;
			this->R = 0;	
		}
	}

	//Empty constructor (black)
	ColorPixel() {
		this->B = 0;
		this->G = 0;
		this->R = 0;	
	}

#pragma endregion

#pragma region Setters and Getters

	//RGB setter
	void SetColor(const uint8_t B, const uint8_t G, const uint8_t R) {
		this->B = B;
		this->G = G;
		this->R = R;
	}

	//Grayscale setter
	void SetColor(const uint8_t grayValue) {
		this->B = grayValue;
		this->G = grayValue;
		this->R = grayValue;
	}

	//RGB getters
	uint8_t GetB() const {
		return B;
	}

	uint8_t GetG() const {
		return G;
	}

	uint8_t GetR() const {
		return R;
	}

#pragma endregion

	//Output operator
	friend std::ostream& operator<<(std::ostream& os, const ColorPixel& c)
	{
	    os << "RGB: (" << static_cast<int>(c.R) << ", " << static_cast<int>(c.G) << ", " << static_cast<int>(c.B) << "); ";
	    return os;
	}

private:
	uint8_t B;
	uint8_t G;
	uint8_t R;
};
