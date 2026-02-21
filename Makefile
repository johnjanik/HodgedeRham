CC = gcc
CFLAGS = -O3 -march=native -Wall -fopenmp
LIBS = -lm -lpng

all: e8_slope_viz e8_f4_viz f4_crystalline_grid exceptional_grid

e8_slope_viz: e8_slope_viz.c e8_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

e8_f4_viz: e8_f4_viz.c e8_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

f4_crystalline_grid: f4_crystalline_grid.c
	$(CC) $(CFLAGS) -o $@ $< -lm

exceptional_grid: exceptional_grid.c e8_common.h
	$(CC) $(CFLAGS) -o $@ $< -lm

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

run-crystal: f4_crystalline_grid
	./f4_crystalline_grid --max-primes 2000000 --n-vertices 38

run-crystal-small: f4_crystalline_grid
	./f4_crystalline_grid --max-primes 10000 --n-vertices 10 --size 1000 --output grid_small.ppm

run-exceptional: exceptional_grid
	./exceptional_grid --max-primes 2000000 --n-vertices 38

run-exceptional-small: exceptional_grid
	./exceptional_grid --max-primes 10000 --n-vertices 10 --size 2000 --output exceptional_small.ppm

run-exceptional-analysis:
	python3 exceptional_analysis.py --max-primes 2000000

run-decoder:
	python3 e8_multi_decoder.py --max-primes 2000000

clean:
	rm -f e8_slope_viz e8_f4_viz e8_decoder e8_f4_analysis f4_crystalline_grid exceptional_grid
