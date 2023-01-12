#include<cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

#include "Structures.h"
#include "Scores.h"
#include "input.h"

//global variables
extern int n_cells;
extern int n_loci;
extern int n_regions;
extern std::vector<Cell> cells;
extern Data data;
extern Params parameters;


void load_CSV(std::string base_name, bool use_CNV, bool apply_filter_regions){
    if (parameters.verbose) {
        std::cout<<"Init data\n\tReading file:" + base_name + "_variants.csv"<<std::endl;
    }
    std::ifstream file_variants(base_name+"_variants.csv");
    if(!file_variants.is_open()) throw std::runtime_error("Could not open variants file");
    // Read region counts (if use CNV)
    std::vector<std::vector<int>> region_counts{};
    data.region_to_name.clear();
    data.region_to_chromosome.clear();
    std::string line, val;
    if(use_CNV){
        std::ifstream file_region(base_name+"_regions.csv");
        if (!file_region.is_open()) throw std::runtime_error("Could not open region file");
        int region_index=0;
        while(std::getline(file_region,line,'\n')){
            std::stringstream ss(line);
            //index: chromosome and name of the region
            std::getline(ss, val, ',');
            int idx_split=0;
            std::vector<int> counts{};
            while (idx_split<val.size() && val[idx_split]!='_') idx_split++;
            if (idx_split<val.size()){
                // First column is chr_regionName
                data.region_to_name.push_back(val.substr(idx_split+1,val.size()-idx_split-1));
                data.region_to_chromosome.push_back(val.substr(0,idx_split));
            }
            else{
                // no chromosome and region_name were given: first column corresponds to the first cell
                data.region_to_name.push_back(std::to_string(region_index));
                data.region_to_chromosome.push_back(std::to_string(region_index));
                counts.push_back(stoi(val));
            }
            while (std::getline(ss, val, ',')){
                counts.push_back(stoi(val));
            }
            region_counts.push_back(counts);
            region_index++;
        }
        n_regions = region_counts.size();
        file_region.close();
    }
    else{
        n_regions=0;
    }

    // Read variants
    data.locus_to_chromosome.clear();
    data.locus_to_position.clear();
    data.locus_to_reference.clear();
    data.locus_to_alternative.clear();
    data.variant_is_SNV.clear();
    data.locus_to_region.clear();
    data.region_to_loci.clear();
    data.region_to_id.clear();
    data.locus_to_name.clear();
    data.locus_to_id.clear();
    data.locus_to_freq.clear();

    std::vector<std::vector<int>> ref_counts{};
    std::vector<std::vector<int>> alt_counts{};
    std::vector<std::vector<int>> genotypes{};
    std::vector<std::string> cell_names{};
    std::vector<int> ref_counts_variant{};
    std::vector<int> alt_counts_variant{};
    std::vector<int> genotypes_variant{};
    
    // First line: header
    std::getline(file_variants,line,'\n'); 
    std::stringstream header(line);
    std::vector<std::string> columns{};
    while (std::getline(header, val, ',')) {
        columns.push_back(val);
        if (val != "CHR" && val != "POS" && val != "REF" && val != "ALT"
                && val != "REGION" && val != "NAME" && val != "FREQ") {
            cell_names.push_back(val);
        }
    }

    // content
    int row_count=0;
    while(std::getline(file_variants,line,'\n')) {
        ref_counts_variant.clear();
        alt_counts_variant.clear();
        genotypes_variant.clear();
        std::stringstream ss(line);
        int column_count = 0;
        std::string locus_id;
        while (std::getline(ss, val, ',')){
            
            if (columns[column_count]=="CHR"){
                data.locus_to_chromosome.push_back(val);
                locus_id += "chr" + val + "_";
            }
            else if (columns[column_count]=="POS"){
                data.locus_to_position.push_back(stoi(val));
                locus_id += val;
            }
            else if (columns[column_count]=="REF"){
                data.locus_to_reference.push_back(val);
            }
            else if (columns[column_count]=="ALT"){
                data.locus_to_alternative.push_back(val);
            }
            else if (columns[column_count]=="REGION"){
                int region_index = -1;
                for (int i = 0; i < data.region_to_name.size(); i++) {
                    if (val == data.region_to_name[i]) {
                        region_index = i;
                    }
                }
                if (region_index == -1) {
                    data.region_to_name.push_back(val);
                    data.region_to_chromosome.push_back(data.locus_to_chromosome[data.locus_to_chromosome.size() - 1]);
                    region_index = n_regions;
                    n_regions++;
                }
                data.region_to_id[val] = region_index;
                data.locus_to_region.push_back(region_index);
            }
            else if (columns[column_count]=="NAME"){
                data.locus_to_name.push_back(val);
            }
            else if (columns[column_count]=="FREQ"){
                data.locus_to_freq.push_back(stod(val));
            }
            else{
                // for each cell, contains RO:AD:GT (:GT being optional)
                int pos = val.find(':');
                ref_counts_variant.push_back(stoi(val.substr(0,pos)));
                int pos2 = val.find(':',pos+1);
                if (pos2==std::string::npos){ // does not contain genotypes
                    alt_counts_variant.push_back(stoi(val.substr(pos+1,val.length()-pos-1)));
                    genotypes_variant.push_back(3);
                }
                else{
                    alt_counts_variant.push_back(stoi(val.substr(pos+1,pos2-pos-1)));
                    genotypes_variant.push_back(stoi(val.substr(pos2+1,1)));
                }
            }
            column_count++;
        }
        ref_counts.push_back(ref_counts_variant);
        alt_counts.push_back(alt_counts_variant);
        genotypes.push_back(genotypes_variant);
        data.locus_to_id[locus_id] = row_count;
        row_count++;
    }
    file_variants.close();

    if (parameters.verbose) {
        std::cout << "\tDone - reading file" << std::endl;
    }

    n_cells = ref_counts[0].size();
    n_loci = ref_counts.size();

    if (data.locus_to_reference.size() == 0) {
        data.variant_is_SNV = std::vector<bool>(n_loci,true);
    } else {
        for (int i = 0; i < n_loci; i++) {
            data.variant_is_SNV.push_back((data.locus_to_reference[i]=="A" || data.locus_to_reference[i]=="C" 
                                            || data.locus_to_reference[i]=="G" || data.locus_to_reference[i]=="T") 
                                &&(data.locus_to_alternative[i]=="A" || data.locus_to_alternative[i]=="C" 
                                            || data.locus_to_alternative[i]=="G" || data.locus_to_alternative[i]=="T"));
        }
    }

    if (data.locus_to_name.size()==0){
        if (data.locus_to_region.size()==n_loci){
            for (int i=0;i<n_loci;i++){
                data.locus_to_name.push_back(data.region_to_name[data.locus_to_region[i]]);
            }
        }
        else{
            for (int i=0;i<n_loci;i++) data.locus_to_name.push_back(std::to_string(i));
        }
    }

    if (use_CNV && data.region_to_name.size()==0){
        for (int k=0;k<n_regions;k++) data.region_to_name.push_back(std::to_string(k));
    }
    
    if (parameters.verbose) {
        std::cout << "\tMapping loci (" << n_loci << " loci) to regions" << std::endl;
    }
    // In case no mapping from variants to regions were provided
    if (data.locus_to_region.size() == 0) {
        if (use_CNV) {
            throw std::invalid_argument("Missing region information for variants. When using CNVs, the variants file must contain a column indicating to which region (amplicon or gene) each variant belongs.");
        }
        for (int i = 0; i < n_loci; i++) {
            data.locus_to_region.push_back(i);
            data.region_to_name.push_back(data.locus_to_name[i]);
            data.region_is_reliable.push_back(false);
            data.region_to_chromosome.push_back(" ");
        }
        n_regions = n_loci;
    }

    if (parameters.verbose) {
        std::cout << "\tMapping regions (" << n_regions << " regions) to loci" << std::endl;
    }
    // Map region index to loci index
    data.region_to_loci.resize(n_regions);
    for (int i = 0; i < n_loci; i++) {
        data.region_to_loci[data.locus_to_region[i]].push_back(i);
    }

    // In case no variant frequencies were provided
    if (data.locus_to_freq.size() == 0) {
        data.locus_to_freq = std::vector<double>(n_loci, 0.0);
    }

    if (parameters.verbose) {
        std::cout << "\tStore by cell (" << n_cells << " cells)" << std::endl;
    }
    // store by cell
    cells.clear();
    cells.reserve(n_cells);
    for (int j = 0; j < n_cells; j++){
        cells.push_back(Cell{});
        cells[j].ref_counts.reserve(n_loci);
        cells[j].alt_counts.reserve(n_loci);
        cells[j].genotypes.reserve(n_loci);
        cells[j].GQ.reserve(n_loci);
        for (int i = 0; i < n_loci; i++) {
            cells[j].ref_counts.push_back(ref_counts[i][j]);
            cells[j].alt_counts.push_back(alt_counts[i][j]);
            cells[j].genotypes.push_back(genotypes[i][j]);
        }
        cells[j].name = cell_names[j];
        if (use_CNV) {
            int total_count = 0;
            cells[j].region_counts.reserve(n_regions);
            for (int k = 0; k < n_regions; k++) {
                cells[j].region_counts.push_back(region_counts[k][j]);
                total_count += region_counts[k][j];
            }
            cells[j].total_counts = total_count;
        }
    }
    
    if (use_CNV) {
        if (apply_filter_regions) {
            filter_regions();
        } else {
            data.region_is_reliable = std::vector<bool>(n_regions, true);
        }
    } else {
        data.region_is_reliable = std::vector<bool>(n_regions, false);
    }

    if (parameters.verbose) {
        std::cout << "Done - Init data" << std::endl;
    }
}





void filter_regions(){
    // Filter out regions for which many cells have 0 (or almost 0) reads
    double threshold = 1.0 / n_regions / 15.0;
    data.region_is_reliable.clear();
    std::string outverbose = "The following regions are excluded from the CNV inference because their coverage is too low: ";
    bool regions_filtered=false;
    for (int k=0;k<n_regions;k++){
        int count_cells_below_threshold=0;
        double mean=0;
        for (int j=0;j<n_cells;j++){
            
            if (1.0*cells[j].region_counts[k] / cells[j].total_counts <= threshold) count_cells_below_threshold++;
            mean+= 1.0*cells[j].region_counts[k] / cells[j].total_counts / n_cells;
        }
        data.region_is_reliable.push_back(((1.0*count_cells_below_threshold/n_cells <= 0.04) && (mean>=0.2/n_regions)));
        regions_filtered = regions_filtered || (!(1.0*count_cells_below_threshold/n_cells <= 0.04) && (mean>=0.2/n_regions));
    }
    if (regions_filtered){
        std::cout<<"The following regions are excluded from the CNV inference because their coverage is too low: ";
        for (int k=0;k<n_regions;k++){
            if (!data.region_is_reliable[k]) std::cout<<data.region_to_name[k]<<",";
        }
        std::cout<<std::endl;
    }
    
}

void init_params(){
    parameters.sequencing_error_rate=0.02; // fixed
	parameters.omega_hom=50.0; // fixed
	parameters.omega_het=8.0; //fixed
	parameters.sequencing_error_rate_indel=0.06; // higher error rate and dispersion for indels because the read calls do not seem so reliable
	parameters.omega_hom_indel = 15.0;
	parameters.omega_het_indel = 4.0;

    parameters.prior_dropoutrate_mean=0.05;
    parameters.prior_dropoutrate_omega=100;

	parameters.theta=6.0;
	parameters.doublet_rate=0.08;

    parameters.use_doublets=true;
    parameters.verbose=true;

    // Tree prior
    parameters.node_cost=1.0;
    parameters.CNLOH_cost=170.0;
    parameters.CNV_cost=85.0;
    parameters.CNV_LOH_cost=85.0;
    parameters.mut_notAtRoot_cost=10;
    parameters.mut_notAtRoot_freq_cost=100000;
}