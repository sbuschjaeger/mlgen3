COMPILER = gcc
FLAGS = -std=c99 -Wall -O3 -lm

all:
	mkdir -p mq_pt_model_binary
	$(COMPILER) $(FLAGS) main.c matquant_pt_c.c -o matquant_pt_c

clean:
	rm -f matquant_pt_c
	rm -rf mq_pt_model_binary
