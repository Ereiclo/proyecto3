
build:
	g++ main.cpp -o exe -std=c++11 -O3 -larmadillo

run: build
	./exe