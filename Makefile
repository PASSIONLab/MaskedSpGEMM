RMATPATH = GTgraph/R-MAT
SPRNPATH = GTgraph/sprng2.0-lite
PRMATPATH = PaRMAT/Release/src
PRMATPATH1 = PaRMAT/Release/

include GTgraph/Makefile.var
INCLUDE += -I$(SPRNPATH)/include
CC = g++

# FLAGS = -g -fopenmp -O3 -march=native
FLAGS = -g -fopenmp -O3 -m64 -march=native -lnuma


# FLAGS = -g -fopenmp -O2 -ffast-math -march=native -ftree-vectorize
sprng:
	(cd $(SPRNPATH); $(MAKE); cd ../..)

rmat:	sprng
	(cd $(RMATPATH); $(MAKE); cd ../..)


prmat:
	(cd $(PRMATPATH1); $(MAKE); cd ../..)

TOCOMPILE = $(RMATPATH)/graph.o $(RMATPATH)/utils.o $(RMATPATH)/init.o $(RMATPATH)/globals.o $(PRMATPATH)/GraphGen_notSorted.o $(PRMATPATH)/utils.o $(PRMATPATH)/Edge.o $(PRMATPATH)/Square.o $(PRMATPATH)/GraphGen_sorted.o

# flags defined in GTgraph/Makefile.var
SAMPLE = ./sample
BIN = ./bin
SRC_SAMPLE = $(wildcard $(SAMPLE)/*.cpp)
SAMPLE_TARGET = $(SRC_SAMPLE:$(SAMPLE)%=$(BIN)%)

spgemm: rmat prmat $(SAMPLE_TARGET:.cpp=_hw)

$(BIN)/%_hw: $(SAMPLE)/%.cpp
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP -DHW_EXE ${TOCOMPILE} ${LIBS}

# specific for OuterSpGEMM
# Will do the same for other exe files
$(BIN)/OuterSpGEMM_hw: outer_mult.h sample/OuterSpGEMM.cpp
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP -DHW_EXE ${TOCOMPILE} ${LIBS}

clean:
	(cd GTgraph; make clean; cd ../..)
	(cd $(PRMATPATH1); make clean; cd ../..)
	rm -rf ./bin/*
	# rm -rf assets/*


gen-er:
	./scripts/gen_er.sh

gen-rmat:
	./scripts/gen_rmat.sh

download:
	./scripts/download.sh
