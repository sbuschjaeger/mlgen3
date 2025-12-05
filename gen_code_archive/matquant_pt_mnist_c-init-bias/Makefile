COMPILER = gcc
FLAGS = -std=c99 -Wall -O3 -lm

all:
	mkdir -p mq_pt_model_binary
	$(COMPILER) $(FLAGS) main.c mq_pt_mlp.c -o mq_pt_mlp

clean:
	rm -f mq_pt_mlp
	rm -rf mq_pt_model_binary
