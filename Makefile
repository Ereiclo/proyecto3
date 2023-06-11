
build:
	g++ main.cpp -o exe -std=c++11 -larmadillo -g

run: build
	./exe