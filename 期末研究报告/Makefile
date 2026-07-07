CXX = g++
HIPCC = hipcc
CXXFLAGS = -O2 -std=c++11
HIPCCFLAGS = -O2 -std=c++11

TARGET = test_gaussian

OBJS = gpu_basic.o gpu_shared.o main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(HIPCC) -o $@ $(OBJS)

gpu_basic.o: gpu_basic.cu common.h
	$(HIPCC) $(HIPCCFLAGS) -c gpu_basic.cu

gpu_shared.o: gpu_shared.cu common.h
	$(HIPCC) $(HIPCCFLAGS) -c gpu_shared.cu

main.o: main.cc common.h serial.h
	$(CXX) $(CXXFLAGS) -c main.cc

clean:
	rm -f *.o $(TARGET)

run: $(TARGET)
	./$(TARGET)