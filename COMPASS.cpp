#include <iostream>
#include <cmath>
#include <cfloat>
#include <random>
#include <fstream>
#include <string.h>
#include <omp.h>

#include "Inference.h"
#include "Tree.h"
#include "Scores.h"
#include "input.h"

int n_cells;
int n_loci;
int n_regions;
std::vector<Cell> cells;
Data data;
Params parameters;

int main(int argc, char* argv[]) {
    init_params();
    parameters.verbose = false;
    // Read command line arguments
    std::string input_file{};
    int n_chains = 4;
    int chain_length = 5000;
    int burn_in = 1000;
    double temperature = 10;
    double betabin_overdisp = parameters.omega_het;
    bool use_CNV = true;
    bool apply_filter_regions = true;
    bool output_simplified = true;
    std::string output{};
    data.sex = "female";
    data.start_tree = "";

    for (int i = 1 ; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            input_file = argv[i+1];
        } else if (strcmp(argv[i], "--nchains") == 0) {
            n_chains = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "--chainlength") == 0) {
            chain_length = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "--burnin") == 0) {
            burn_in = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "--temperature") == 0) {
            temperature = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "--overdisp") == 0) {
            betabin_overdisp = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-o") == 0) {
            output = argv[i+1];
        } else if (strcmp(argv[i], "-d") == 0 && strcmp(argv[i+1], "0") == 0) {
            parameters.use_doublets = false;
        } else if (strcmp(argv[i], "--doubletrate") == 0) {
            parameters.doublet_rate = atoi(argv[i+1]);
        } else if (strcmp(argv[i], "--CNV") == 0 && strcmp(argv[i+1], "0") == 0) {
            use_CNV = false;
        } else if (strcmp(argv[i], "--filterregions") == 0 && strcmp(argv[i+1], "0") == 0) {
            apply_filter_regions = false;
        } else if (strcmp(argv[i], "--sex") == 0) {
           data.sex = std::string(argv[i+1]);
        } else if (strcmp(argv[i], "--tree") == 0) {
           data.start_tree = std::string(argv[i+1]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
           parameters.verbose = true;
           n_chains = 1;
        } else if (strcmp(argv[i], "--prettyplot") == 0 && strcmp(argv[i+1], "0") == 0) {
            output_simplified = false;
        }
    }
    if (output == "") {
        std::cout << "!ERROR: -o argument has to be specified!" << std::endl;
        return 1;
    }

    load_CSV(input_file, use_CNV, apply_filter_regions);

    parameters.omega_het = std::min(parameters.omega_het, betabin_overdisp);
    parameters.omega_het_indel = std::min(parameters.omega_het_indel, betabin_overdisp);

    // Get the name of the file, without directory
    std::string input_name = input_file;
    int name_start = 0;
    for (int i = 0; i < input_file.size(); i++) {
        if (input_file[i] == '/') {
            name_start = i + 1;
        }
    }

    input_name = input_file.substr(name_start, input_file.size() - name_start);

	std::vector<double> results{};
    results.resize(n_chains);
    std::vector<Tree> best_trees{};
    best_trees.resize(n_chains); // Constructor called twice here: not pretty
    std::vector<Inference> chains{};
    chains.resize(n_chains); // Constructor called twice here: not pretty
    if (n_chains < omp_get_num_procs()) {
        omp_set_num_threads(n_chains);
    } else {
        omp_set_num_threads(omp_get_num_procs());
    }

    if (parameters.verbose) {
        std::cout << "Starting 1 MCMC chain" << std::endl;
        if (data.start_tree != "") {
            chains[0] = Inference{data.start_tree, "", temperature, 0};
        } else {
            chains[0] = Inference{"", temperature, 0};
        }
        std::cout << "MCMC - running" << std::endl;
        best_trees[0] = chains[0].find_best_tree(use_CNV, chain_length, burn_in);
        std::cout << "MCMC - done" << std::endl;
        results[0] = best_trees[0].log_score;
        std::cout << "Saving results - done" << std::endl;
    } else {
        std::cout << "Starting " << std::to_string(n_chains) << " MCMC chains in parallel" << std::endl;
        // run in parallel without initializing the vectors
        // https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
        #pragma omp parallel for
    	for (int i = 0; i < n_chains; i++) {
    		std::srand(i);
            if (data.start_tree != "") {
    		    chains[i] = Inference{data.start_tree, "", temperature, i};
            } else {
                chains[i] = Inference{"", temperature, i};
            }
            best_trees[i] = chains[i].find_best_tree(use_CNV, chain_length, burn_in);
    		results[i] = best_trees[i].log_score;
    	}
    }
    std::cout << "MCMC chains - done" << std::endl;

    double best_score = -DBL_MAX;
    int best_score_index = -1;
	for (int i = 0; i < n_chains; i++) {
		if (best_score < results[i]) {
            best_score = results[i];
            best_score_index = i;
        }
	}
    if (output_simplified) {
        best_trees[best_score_index].to_dot_pretty(output);
    } else {
        best_trees[best_score_index].to_dot(output);
    }

    std::string gv_filename(output);
    if (output.substr(output.size() - 3) != ".gv") {
        gv_filename = output + "_tree.gv";
    }
    std::cout << "Completed! The output was written to " << output << ". You can visualize the tree by running: dot -Tpng " << gv_filename << " -o output.png" << std::endl;
	return 0;
}