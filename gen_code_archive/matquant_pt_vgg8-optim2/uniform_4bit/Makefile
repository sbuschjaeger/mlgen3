COMPILER = g++
FLAGS = -std=c++17 -Wall -O3 -funroll-loops -ftree-vectorize

# Extract model binary path and directory
MODEL_NAME = $(shell basename models/matquant/mnist_pt/mq_pt_model.pt .pt)
BINARY_DIR = $(MODEL_NAME)_binary

all:
	# Create binary directory if it doesn't exist
	mkdir -p $(BINARY_DIR)
	# Copy PT model file if it exists
	if [ -f "models/matquant/mnist_pt/mq_pt_model.pt" ]; then cp "models/matquant/mnist_pt/mq_pt_model.pt" . 2>/dev/null || true; fi
	# Compile the model
	$(COMPILER) $(FLAGS) main.cpp matquant_pt_vgg8_uniform_4bit.cpp -o matquant_pt_vgg8_uniform_4bit
