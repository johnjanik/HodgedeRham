CC = gcc
CFLAGS = -O3 -march=native -Wall -fopenmp
LIBS = -lm -lpng

all: e8_slope_viz e8_f4_viz

e8_slope_viz: e8_slope_viz.c e8_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

e8_f4_viz: e8_f4_viz.c e8_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# --- Run targets ---

run-viz: e8_slope_viz
	./e8_slope_viz --max-primes 100000000 --dpi 1200 --size 24

run-f4: e8_f4_viz
	./e8_f4_viz --max-primes 100000000 --dpi 1200 --size 16

run-f4-2M: e8_f4_viz
	./e8_f4_viz --max-primes 2000000 --dpi 1200 --size 16

run-viz-10k: e8_slope_viz
	./e8_slope_viz --max-primes 10000 --dpi 300 --size 10 --csv

run-viz-1M: e8_slope_viz
	./e8_slope_viz --max-primes 1000000 --dpi 600 --size 16

clean:
	rm -f e8_slope_viz e8_f4_viz e8_decoder e8_f4_analysis
