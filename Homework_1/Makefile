CXX = g++
CXXFLAGS = -Wall -g
INCLUDES = -I/usr/local/include/opencv4
LFLAGS = -L/usr/local/lib
LIBS = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)
MAIN = helloWorld

.PHONY: clean

# Target rules here
all: $(MAIN)
	@echo Simple compiler named helloWorld has been compiled

$(MAIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)