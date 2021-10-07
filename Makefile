solver: Source.cpp
	g++ -std=c++1z -fopenmp -static-libstdc++ -O2 -flto -march=native Source.cpp -o solver
clean:
	rm -f *.o solver