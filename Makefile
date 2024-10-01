CC = g++
CCSW = -O3 -Wno-deprecated-declarations

# Set the source directory
SRC_DIR = lander_src

# Define the object files
OBJS = $(SRC_DIR)/lander.o $(SRC_DIR)/lander_graphics.o $(SRC_DIR)/lander_mechanics.o $(SRC_DIR)/autopilot.o $(SRC_DIR)/main.o

all: lander

lander: $(OBJS)
	$(CC) -o $@ $^ ${CCSW} -lGL -lGLU -lglut
	@echo Linking for Linux

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/lander.h
	$(CC) ${CCSW} -c $< -o $@

clean:
	@echo cleaning up
	@rm -f $(SRC_DIR)/*.o lander