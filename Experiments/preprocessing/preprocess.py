import argparse
import os
import re
import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
from sklearn.mixture import GaussianMixture
import loompy
from varcode import Variant
from pyensembl import ensembl_grch37, ensembl_grch38


# Exclude some sites with high error rates
blacklist = ["1_43815093","1_115256626","2_25463686","2_25467178","2_25469567","2_25458738","2_209113336","2_25536827","2_198267672","2_209113332","4_106194088","4_106157187","4_106196675","4_106196287","7_148504716",\
    "7_148506185","7_148506191","7_148504854","7_148506194","7_148526908","7_148504717","7_148543582","7_148543583","8_128750698","8_117864842",\
        "12_25378673","12_25378676","13_28592669","13_28602256","17_29559928","17_29562734","17_7578587",\
    "17_7579801","17_29559932","17_7579414","17_29483195","17_29559926","17_29562734","17_7579440","20_31024389",\
    "21_44524505","21_36259324","X_133527541", "X_44911052",\
    "X_39932806","X_39932807","X_44929002","X_15809170","X_39922359","X_15821932","X_15841334","X_15838366","X_15841334","X_15841336",\
    "X_39932907"," X_53426504","X_133549184","X_53426504","X_39914742","X_53426570","X_44949032","X_39921505","X_15827406",\
        "17_7578115","17_7579472"]
#1_43815093 for sample AML-91-001
#1_115256626 for sample AML-83-001
#2_25467178 for sample AML-99-001
#2_25469567 for AML-107-001
#2_25458738 for AML-46-001
#2_209113336 for AML-112-001
#2_25536827 for AML-114-001
#4_106194088 for sample AML-83-001
#4_106157187 for AML-42-001
#4_106196675 for AML-42-001
#7_148504716 for AML-23-001 and AML-27-001
#7_148506185 for AML-95-001
#7_148506191 for AML-99-002
#7_148504854 for AML-22-001
#7_148506194 for AML-98-001
#7_148526908 for AML-98-001
#7_148504717 for AML-38-003
#7_148543582 and 7_148543582 for AML-107-001
#13_28592669 for AML-107-001
#13_28592546 for AML-113-001; but useful for AML-03!
#17_7579801 for AML-83-001
#17_29559928 for AML-116-001
#17_29562734 for AML-117-001
#17_29559932 for AML-109-001
#20_31024389 for AML-91-001
#21_36259324 for AML-114-001
# X_133527541 for AML-93-001
# X_44911052 for AML-95-001
# X_39932806 and X_39932807 for AML-91-001
# X_44929002 for AML-85-001
# X_15809170 for AML-91-001
# X_39922359 for AML-05-001
# X_15821932 for AML-84-001
# X_15841334 for AML-107-001
# X_39932907 for AML-109-001
# X_39933339 for AML-109-001
# X_53426504 for AML-118-001




def get_1K_freq(file,chr,pos,ref,alt):
    """
    Parse the 1000Genomes VCF to get the population frequency of a variant.
    Start from the last position, to avoid iterating several times through the file (the variants are sorted)
    """
    current_chr = "0"
    current_pos = 0
    line = True
  
    last_pos_file = file.tell()
    pos_file = file.tell()
    while line and current_chr!=chr: # get to the right chromosome
        last_pos_file = file.tell()
        pos_file = file.tell()
        line = file.readline()
        if line and line[0]!="#":
            linesplit = line.split("\t")
            current_chr = linesplit[0]
            current_pos = int(linesplit[1])
  
    while line and current_chr==chr and current_pos<pos:
        last_pos_file = pos_file
        pos_file = file.tell()
        line = file.readline()
        if line:
            linesplit = line.split("\t")
            current_chr = linesplit[0]
            current_pos = int(linesplit[1])
    file.seek(last_pos_file) # go back one line (useful at the limit between two chromosomes.)
    if pos==current_pos:
        ref_SNP = linesplit[3]
        AFs = linesplit[7][3:].split(",")
        alts = linesplit[4].split(",")
        for index_alt in range(len(alts)):
            if alts[index_alt] == alt and ref_SNP==ref:
                return float(AFs[index_alt])
    
    return 0

def get_gnomad_freq(file,chr,pos,ref,alt):
    """
    Parse the gnomAD csv to get the population frequency of a variant.
    Start from the last position, to avoid iterating several times through the file (the variants are sorted)
    """
    current_chr = "0"
    current_pos = 0
    line = True
  
    last_pos_file = file.tell()
    pos_file = file.tell()
    while line and current_chr!=chr: # get to the right chromosome
        last_pos_file = file.tell()
        pos_file = file.tell()
        line = file.readline()
        if line:
            linesplit = line.split(",")
            current_chr = linesplit[0]
            current_pos = int(linesplit[1])
  
    while line and current_chr==chr and current_pos<pos:
        last_pos_file = pos_file
        pos_file = file.tell()
        line = file.readline()
        if line:
            linesplit = line.split(",")
            current_chr = linesplit[0]
            current_pos = int(linesplit[1])
    file.seek(last_pos_file) # go back one line (useful at the limit between two chromosomes.)

    if pos==current_pos:
        ref_SNP = linesplit[3]
        AF = linesplit[7]
        alt_SNP = linesplit[4]
        if alt_SNP == alt and ref_SNP==ref:
            return float(AF)
    return 0

def correct_amplicon_artefacts(amplicon_matrix):
    # In Tapestri data, it appears that often there are 2 clusters of cells which have very different distributions for the amplicon read counts.
    # This seems to be an artefact, because this clustering appears to be independant from the mutations of the cells.
    n_amplicons,n_cells = amplicon_matrix.shape
    total_counts = np.sum(amplicon_matrix,axis=0)
    amplicon_proportions = np.zeros((n_cells,n_amplicons))
    for j in range(n_cells):
        for k in range(n_amplicons):
            amplicon_proportions[j,k] = amplicon_matrix[k,j] / total_counts[j]
    gm = GaussianMixture(n_components=2, random_state=0).fit(amplicon_proportions)
    probs = gm.predict_proba(amplicon_proportions)
    
    differences = []
    for k in range(n_amplicons):
        differences.append(gm.means_[1][k] - gm.means_[0][k])

    corrected_counts = np.zeros((n_amplicons,n_cells))
    for j in range(n_cells):
        direction = 1.0 - 2*probs[j][1]
        for k in range(n_amplicons):
            corrected_counts[k,j] = max(0,round((amplicon_proportions[j,k] + differences[k]/2.0 * direction)*total_counts[j]))
            
    return corrected_counts
    

def convert_loom(infile,outdir, min_GQ, min_DP, min_AF, min_frac_cells_genotyped, min_frac_loci_genotyped,min_frac_cells_alt, region="gene",\
                    correct_read_counts=True,reference=37,verbose=True,SNP_file=None,panel_file=None,whitelist_file=None):
    if reference==37:
        ensembl_ref = ensembl_grch37
    else:
        ensembl_ref = ensembl_grch38
    basename = os.path.basename(infile)[:-5]
    if SNP_file != None:
        SNP_f = open(SNP_file,"r")

    # If selected mutations were already provided
    whitelist=[]
    whitelist_positions=[]
    if whitelist_file is not None:
        df_mutations = pd.read_csv(whitelist_file)
        for i in df_mutations.index:
            sample = df_mutations.loc[i,"sample ID"]
            if sample!=basename: continue
            pos = str(df_mutations.loc[i,"chr"])+"_"+str(df_mutations.loc[i,"start"])
            whitelist_positions.append(pos)
            mutation = pos + "_"+df_mutations.loc[i,"ref allele"] + "_"+df_mutations.loc[i,"alt allele"]
            whitelist.append(mutation)

    with loompy.connect(infile) as ds:
        n_loci_full,n_cells_full = ds.shape
        chromosomes_full = ds.ra["CHROM"]
        positions_full = ds.ra["POS"]
        # The loom files appear to not always be sorted. (or partially sorted...)
        # reorder loci by position and chromosome
        chr_order = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","X","Y"]
        for chrom in chromosomes_full:
            chr = str(chrom)
            if not chr in chr_order:
                print("WARNING: chromosome "+ str(chr) + " was not recognized.")
        loci_by_chr={}
        for i in range(n_loci_full):
            chr = str(chromosomes_full[i])
            if not chr in loci_by_chr:
                loci_by_chr[chr]=[]
            loci_by_chr[chr].append((positions_full[i],i))
        sorted_loci=[]
        for chr in chr_order:
            if chr in loci_by_chr:
                loci_by_chr[chr].sort()
                for x in loci_by_chr[chr]:
                    sorted_loci.append(x[1])

        #amplicons
        amplicons_full = ds.ra["amplicon"][:]

        #get list of unique amplicon names
        amplicon_names=[]
        amplicon_chromosomes=[]
        if not panel_file is None:
            df_panel = pd.read_csv(panel_file)
        amplicon_map={}
        for i in sorted_loci:
            if amplicons_full[i]!="nan" and amplicons_full[i]!="":
                if len(amplicon_names)==0 or amplicons_full[i]!=amplicon_names[-1]:
                    amplicon_name = amplicons_full[i]
                    if not panel_file is None:
                        for k in range(df_panel.shape[0]):
                            if df_panel.loc[k,"Chr"]==chromosomes_full[i] and df_panel.loc[k,"Primer Start"]<=positions_full[i] \
                                and df_panel.loc[k,"Primer End"]>=positions_full[i]:
                                amplicon_name = df_panel.loc[k,"Amplicon"]
                                continue
                    amplicon_map[amplicons_full[i]] = amplicon_name
                    amplicon_names.append(amplicons_full[i])
                    amplicon_chromosomes.append(chromosomes_full[i])
        n_amplicons = len(amplicon_names)
    
        

        genotypes = ds[:,:]
        # First select loci where the alt allele is present in some cells, to avoid loading the 5 full matrices in memory
        candidate_loci = []
        for i in sorted_loci:
            chr_pos = str(chromosomes_full[i])+"_"+str(positions_full[i])
            if chr_pos in blacklist: continue
            n_alt = np.sum( (genotypes[i,:]==1) | (genotypes[i]==2) )
            if n_alt / n_cells_full >= min_frac_cells_alt or chr_pos in whitelist_positions:
                candidate_loci.append(i)
        del genotypes
        print("Number of candidate loci: " + str(len(candidate_loci)))

        sorted_candidate = sorted(candidate_loci)
        argsort_candidate = np.argsort(candidate_loci)
        reverse_sort = [0]*len(candidate_loci)
        for i in range(len(candidate_loci)):
            reverse_sort[argsort_candidate[i]] = i

        # Load only the data for the candidate loci. The array has to be loaded with sorted indices
        chromosomes = (ds.ra["CHROM"][sorted_candidate])[reverse_sort]
        positions = (ds.ra["POS"][sorted_candidate])[reverse_sort]
        amplicons = (ds.ra["amplicon"][sorted_candidate])[reverse_sort]
        ref = (ds.ra["REF"][sorted_candidate])[reverse_sort]
        alt = (ds.ra["ALT"][sorted_candidate])[reverse_sort]
        DP = (ds.layers["DP"][sorted_candidate,:])[reverse_sort,:]
        AD = (ds.layers["AD"][sorted_candidate,:])[reverse_sort,:]
        RO = (ds.layers["RO"][sorted_candidate,:])[reverse_sort,:]
        GQ = (ds.layers["GQ"][sorted_candidate,:])[reverse_sort,:]
        genotypes = (ds[sorted_candidate,:])[reverse_sort,:]
        
        # Set low quality genotypes to "missing"
        prefiltered_loci = []
        lowqual_genotypes = (GQ<min_GQ) | (DP<min_DP) | ( (genotypes!=0)& (AD/(DP+0.1)<min_AF) )
        genotypes[lowqual_genotypes] = 3
        # Keep loci which are genotyped in at least min_frac_cells_genotyped of the cells
        for i in range(len(candidate_loci)):
            count_cells_genotyped = np.sum(genotypes[i,:]!=3)
            v = str(chromosomes[i])+"_"+str(positions[i])+"_"+ref[i]+"_"+alt[i]
            if count_cells_genotyped/n_cells_full>min_frac_cells_genotyped or v in whitelist: 
                prefiltered_loci.append(i)

        # Keep cells for which at least X% of the variants are genotyped
        filtered_cells = []
        for j in range(n_cells_full):
            count_loci_genotyped = np.sum(genotypes[prefiltered_loci,j] !=3)
            if count_loci_genotyped/len(prefiltered_loci)>= min_frac_loci_genotyped:
                filtered_cells.append(j)
        n_cells = len(filtered_cells)
 
        # Filter loci
        filtered_loci=[]
        variant_names=[]
        variant_frequencies=[]
        last_locus=-1
        lastvariant_nonsynonymous = True
        for i in prefiltered_loci:
            count_cells0 = np.sum(genotypes[i,:]==0)
            count_cells1 = np.sum(genotypes[i,:]==1)
            count_cells2 = np.sum(genotypes[i,:]==2)
            print(amplicons[i]+": "+ str(positions[i])+ ": "+str(ref[i])+","+str(alt[i]) +": "+str(count_cells0)+","+str(count_cells1)+","+str(count_cells2))
            v = str(chromosomes[i])+"_"+str(positions[i])+"_"+ref[i]+"_"+alt[i]
            if not v in whitelist:
                # Filter out loci with very few cells genotyped as alt. The filter is a bit more restrictive when there are few total cells
                if (count_cells1+count_cells2)/max(4000,len(filtered_cells))<=min_frac_cells_alt: continue
                if (np.sum(AD[i,:]>min_DP)/max(4000,len(filtered_cells))<=min_frac_cells_alt): continue # require minimum AD for evidence of alt
                print(np.sum(AD[i,:]>min_DP))
                # Filter out germline variants homozygous alt
                #if count_cells0 <=0.001 * count_cells2 and count_cells1<0.03*count_cells2: continue
                if count_cells0 <=0.01 * max(4000,n_cells) and count_cells1<0.03*max(4000,n_cells): continue
                print("Passed first filters")

            
            

            # Get the effect of the variant on the protein (if coding mutation) and its frequency in the population
            # The filters are less restrictive for non-synonymous mutations which have a low frequency in the population
            if amplicons[i]=="FLT3_2_a3" and len(alt[i])>1: # Handle FL3-ITD separately
                nonsynonymous_variant=True
                variant_name = "FLT3-ITD"
            else:
                if ref[i]=="*": ref[i] = ""
                if alt[i]=="*": alt[i] = ""
                if ("+" in alt[i]):
                    alt_split=alt[i].split("+")
                    ref[i] = alt_split[0]
                    alt[i] = alt_split[1]
                variant = Variant(contig=str(chromosomes[i]), start=positions[i], ref=ref[i], alt=alt[i], ensembl=ensembl_ref)
                effects = variant.effects()
                topPriorityEffect = effects.top_priority_effect()
                nonsynonymous_variant = (topPriorityEffect.gene_name is not None) and (not topPriorityEffect.short_description in ["silent","intronic","3' UTR","5' UTR","incomplete"])
                
                if topPriorityEffect.gene_name is not None:
                    variant_name = topPriorityEffect.gene_name + " " + topPriorityEffect.short_description
                else:
                    variant_name = amplicons[i] + " intergenic"

            # Frequency of the variant in the population
            if SNP_file == None:
                variant_frequency = 0
            else:
                variant_frequency = get_1K_freq(SNP_f,str(chromosomes[i]),int(positions[i]),str(ref[i]),str(alt[i]))
                #variant_frequency = get_gnomad_freq(SNP_f,str(chromosomes[i]),int(positions[i]),str(ref[i]),str(alt[i]))
               

            # Stricter filters for SNPs and silent mutations
            if variant_frequency>0.0001:
                max_dropout = 1.0 #0.09
                max_ratio_dropoutallele = 1.5 #1.4
                min_cells_hom = 120
            elif not nonsynonymous_variant:
                max_dropout = 0.35
                max_ratio_dropoutallele=1.3 #1.2
                min_cells_hom=100
            else:
                max_dropout = 0.08
                max_ratio_dropoutallele=1.1
                min_cells_hom=50
            if not v in whitelist:
                #Higher min frequency for common SNPs and variants with no protein effect
                if (variant_frequency>0.0001 or not nonsynonymous_variant) and (count_cells1+count_cells2)/len(filtered_cells)<=2*min_frac_cells_alt: continue
                # Stricter filters for homozygous alt for SNPs
                if (variant_frequency>0.0001 or not nonsynonymous_variant) and (count_cells0+count_cells1)<0.07*max(3000,n_cells): continue

                # Indels are often unreliable
                is_small_indel = len(ref[i])!=1 or (len(alt[i])!=1 and len(alt[i])<=6)
                if (is_small_indel):
                    if ((not nonsynonymous_variant) or (count_cells1+count_cells2)/max(4000,len(filtered_cells))<=2*min_frac_cells_alt) and n_amplicons>150:
                        if verbose:
                            print("Filtered out an indel in amplicon " + str(amplicons[i]) +" because it had no protein impact")
                        continue
                    max_ratio_dropoutallele=max(max_ratio_dropoutallele,1.3)
                    max_dropout = max(max_dropout,0.10)
                    min_cells_hom=max(min_cells_hom,60)
                    

                # Filter out germline heterozygous variants which are not lost in any cells
                if (max(count_cells0,count_cells2)<max_dropout * count_cells1 and 40+count_cells2 < max_ratio_dropoutallele* (40+count_cells0) \
                    and (40+count_cells0 < max_ratio_dropoutallele*(40+count_cells2) or is_small_indel) ) \
                    or max(count_cells0,count_cells2)<min_cells_hom:
                    if verbose:
                        print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because it seems to be a germline variant not lost in any cells: "\
                            +str(count_cells0) + " hom ref, " + str(count_cells1) + " het, " + str(count_cells2) + " hom alt.")
                    continue
                        
                # Remove loci for which one allele has always less than 28% of the reads, when the allele is detected (but still allow some sequencing errors)
                # They probably correspond to sequencing/mapping errors.
                threshold = 0.28
                if nonsynonymous_variant: threshold = 0.20
                if nonsynonymous_variant and n_amplicons>100: threshold=0.15
                count_reffreq_above = count_reffreq_below = count_altfreq_below = count_altfreq_above=0
                for j in filtered_cells:
                    if AD[i,j]>0 and AD[i,j]>0.03*RO[i,j]:
                        if AD[i,j]/DP[i,j]>threshold: count_altfreq_above+=1
                        else: count_altfreq_below+=1
                    if RO[i,j]>0 and RO[i,j]>0.03*AD[i,j]:
                        if RO[i,j]/DP[i,j]>threshold: count_reffreq_above+=1
                        else: count_reffreq_below+=1
                print(str(count_altfreq_above)+","+str(count_altfreq_below)+","+str(count_reffreq_above)+","+str(count_reffreq_below))
                if (count_altfreq_above<count_altfreq_below and count_altfreq_above+count_altfreq_below>500) \
                    or (count_reffreq_above<count_reffreq_below and count_reffreq_above+count_reffreq_below>500):
                    if amplicons[i]!="FLT3_2_a3": # exclude FLT3-ITD from this filter
                        if verbose: print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because one allele always had a higher frequency than the other.")
                        continue
        
                # Filter out deletions (their genotyping seems imprecise)
                if (alt[i]=="*" or (len(alt[i])<len(ref[i]))) and variant_frequency>0.0001 :
                    if verbose: print("Filtered out a locus in amplicon"+str(amplicons[i])+" because it is a common deletion.")
                    continue

                # Filter out variants for which the sequencing depth is always low when the variant is present
                depth_wt = np.mean(DP[i,(genotypes[i,:]==0)])
                depth_mut = np.mean(DP[i,((genotypes[i,:]==1) | (genotypes[i,:]==2) )])
                max_ratio = 0.1 if nonsynonymous_variant else 0.25
                if depth_mut<max_ratio * depth_wt:
                    if verbose: print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because the sequencing depth was low when the mutation was detected.")
                    continue
            
                # Only keep one locus per amplicon, except if two loci have different genotypes in different cells and both seem to be somatic variants
                if last_locus!=-1 and amplicons[i]==amplicons[last_locus]:
                    count_same_genotype=0
                    count_opposite_genotype=0
                    count_available_genotype=0
                    for j in filtered_cells:
                        gen1 = genotypes[i,j]
                        gen2 = genotypes[last_locus,j]

                        #if (gen1==1 or gen1==2) and (gen2==1 or gen2==2): count_both_mut+=1
                        #if (gen1==1 or gen2==1 or gen1==2 or gen2==2) and gen1!=3 and gen2!=3: count_mut+=1

                        if gen1==gen2 and gen1!=3: count_same_genotype+=1
                        if (gen1==0 and gen2==2) or (gen1==2 and gen2==0) or (gen1==1 and gen2==1): count_opposite_genotype+=1
                        if (gen1!=3 and gen2!=3): count_available_genotype+=1
                    similarity = max(count_same_genotype,count_opposite_genotype)/count_available_genotype
                    #similarity2 = count_both_mut / (0.01+count_mut)
                    
                    if variant_frequency>=0.001: # this variant is a SNP: ignore it 
                        if verbose: print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because there were several variants in the same amplicon.")
                        continue
                    elif len(variant_frequencies)>0 and variant_frequencies[-1]>=0.001: # the previous variant was a SNP: remove it
                        if verbose: print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because there were several variants in the same amplicon.")
                        filtered_loci.pop()
                        variant_names.pop()
                        variant_frequencies.pop()
                    elif similarity>0.97: # both variants have the same genotype in the same cells: can ignore one
                        if verbose: print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because there were several variants in the same amplicon.")
                        continue
                    elif (not nonsynonymous_variant) and similarity>0.80: # this variant is intergenic/intronic/synonymous: more likely to ignore it
                        if verbose: print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because there were several variants in the same amplicon.")
                        continue
                    elif len(variant_frequencies)>0 and (not lastvariant_nonsynonymous) and similarity>0.80: # the last variant is intergenic/intronic/synonymous: more likely to remove it
                        if verbose: print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because there were several variants in the same amplicon.")
                        filtered_loci.pop()
                        variant_names.pop()
                        variant_frequencies.pop()

            # Merge when one FLT3-ITD is split into several lines in the loom file (seems to happen when some reads don't go through the full ITD)
            if len(filtered_loci)>0:
                i2 = filtered_loci[-1]
                if chromosomes[i]==chromosomes[i2] and abs(positions[i]-positions[i2])<=max(len(alt[i]),len(alt[i2])) and len(alt[i])>4 and len(alt[i2])>4:
                    if alt[i] in alt[i2] or alt[i2] in alt[i]:
                        AD[i2,:] = AD[i2,:] + AD[i,:]
                        for j in filtered_cells:
                            RO[i2,j] = max(RO[i2,j],RO[i,j])
                            if (genotypes[i,j] in [1,2]): genotypes[i2,j] = genotypes[i,j]
                        if verbose:  print("Filtered out a locus in amplicon " + str(amplicons[i]) + " because several insertions were similar.")
                        continue


            filtered_loci.append(i)
            variant_names.append(variant_name)
            variant_frequencies.append(variant_frequency)
            last_locus=i
            lastvariant_nonsynonymous = nonsynonymous_variant
            if verbose: print("** Selected variant " + str(variant_name) + ", "+str(chromosomes[i])+"_"+str(positions[i])+ " (freq "+str(variant_frequency)+"): "\
                +str(count_cells0) + " hom ref, " + str(count_cells1) + " het, " + str(count_cells2) + " hom alt.")
        n_loci = len(filtered_loci)

        variants_info = {"CHR":[],"POS":[],"REF":[],"ALT":[],"REGION":[],"NAME":[],"FREQ":[]}
        for j in range(n_cells):
            variants_info[str(j)]=[]

        for i,locus in enumerate(filtered_loci):
            variants_info["CHR"].append(chromosomes[locus])
            variants_info["POS"].append(positions[locus])
            variants_info["REF"].append(ref[locus])
            variants_info["ALT"].append(alt[locus])
            if region=="gene":
                variants_info["REGION"].append(amplicon_map[amplicons[locus]].split("_")[0])
            elif region=="amplicon":
                variants_info["REGION"].append(amplicon_map[amplicons[locus]])
            else: 
                raise NameError("Region must be gene or amplicon")
            variants_info["NAME"].append(variant_names[i])
            if SNP_file is not None: variants_info["FREQ"].append(variant_frequencies[i])
            else: variants_info["FREQ"].append(0)
            for j in range(n_cells):
                if (GQ[filtered_loci[i],filtered_cells[j]]>0.1):
                    variants_info[str(j)].append(str(int(RO[filtered_loci[i],filtered_cells[j]])) + ":" + str(int(AD[filtered_loci[i],filtered_cells[j]])) \
                                                + ":" + str(int(genotypes[filtered_loci[i],filtered_cells[j]])))
                else: # if genotype quality is 0, consider that we have no reads, as the reads are most likely unreliable
                    variants_info[str(j)].append("0:0:3")
        df_variants = pd.DataFrame(variants_info)
        df_variants.to_csv(os.path.join(outdir,basename+"_variants.csv"),index=False,header=True)


        # Get amplicon read counts for each cell
        amplicon_matrix = np.zeros((n_amplicons,n_cells),dtype=int)

        for k in range(n_amplicons):
            amplicon = amplicon_names[k]
            loci_in_amplicon=[]
            for i in range(n_loci_full):
                if amplicons_full[i]==amplicon:
                    loci_in_amplicon.append(i)
            DP_amplicon = ds.layers["DP"][loci_in_amplicon,:]
            for j in range(n_cells):
                amplicon_matrix[k,j] = np.median(DP_amplicon[:,filtered_cells[j]])

        

        if correct_read_counts:
            amplicon_matrix = correct_amplicon_artefacts(amplicon_matrix)
        # index: chromosome and name of the amplicons
        index_amplicons=[str(amplicon_chromosomes[k])+"_"+amplicon_map[amplicon_names[k]] for k in range(n_amplicons)]
        df_amplicons = pd.DataFrame(amplicon_matrix,index=index_amplicons,dtype=int)
        if region=="amplicon":
            df_amplicons.to_csv(os.path.join(outdir,basename+"_regions.csv"),sep=",",header=False,index=True)

        amplicon_to_gene = []
        gene_to_amplicons = []
        gene_to_name = []
        gene_to_chr = []
        n_genes=0
        for k in range(n_amplicons):
            amplicon = amplicon_map[amplicon_names[k]]
            gene = amplicon.split("_")[0] # Assume amplicon name is gene_xx
            if gene in gene_to_name:
                g = gene_to_name.index(gene)
                amplicon_to_gene.append(g)
                gene_to_amplicons[g].append(k)
            else:
                g = n_genes
                n_genes+=1
                amplicon_to_gene.append(g)
                gene_to_amplicons.append([k])
                gene_to_name.append(gene)
                gene_to_chr.append(amplicon_chromosomes[k])
        
        gene_matrix = np.zeros((n_genes,n_cells))
        for g in range(n_genes):
            for j in range(n_cells):
                for k in gene_to_amplicons[g]:
                    gene_matrix[g,j]+=amplicon_matrix[k,j]
        index_genes=[str(gene_to_chr[g])+"_"+gene_to_name[g] for g in range(n_genes)]
        df_genes = pd.DataFrame(gene_matrix,index=index_genes,dtype=int)
        if region=="gene":
            df_genes.to_csv(os.path.join(outdir,basename+"_regions.csv"),sep=",",header=False,index=True)
            
    

    if SNP_file != None:
        SNP_f.close()

parser = argparse.ArgumentParser()
parser.add_argument('-i', type = str, help='Input. Can be a loom file or a directory of loom files')
parser.add_argument('-o', type = str, help='Output directory')
parser.add_argument('--SNP', type = str,default = None, help='File containing the frequencies of SNPs in the population')
parser.add_argument('--region', type = str,default = "gene", help='Which region to use for CNVs (gene or amplicon)')
parser.add_argument('--panel', type = str,default = None, help='CSV metadata file for the amplicons')
parser.add_argument('--whitelist', type = str,default = None, help='CSV file containing the mutations to always include')
parser.add_argument('--ref', type = int,default = 37, help='Reference genome (37 or 38)')
parser.add_argument('--correct', dest='correct', default=False,action='store_true')
args = parser.parse_args()

if len(args.i)>5 and args.i[-5:]==".loom":
    convert_loom(args.i,args.o,15,6,0.2,0.25,0.4,0.015,region=args.region,correct_read_counts = args.correct, reference=args.ref,verbose=True,SNP_file=args.SNP,panel_file = args.panel, whitelist_file = args.whitelist)
else:
    for f in sorted(os.listdir(args.i)):
        if len(f)>5 and f[-5:]==".loom":
            convert_loom(os.path.join(args.i,f),args.o,15,6,0.2,0.25,0.4,0.015,region=args.region,reference=args.ref,verbose=True,SNP_file=args.SNP,panel_file = args.panel,whitelist_file=args.whitelist)



#convert_loom("../loom_AML/AML-50-001.loom","../preprocessed_data",15,5,0.2,0.3,0.5,0.02,reference=37,verbose=True,SNP_VCF="../genome1K.phase3.SNP_AF5e2.chr1toX.hg19.vcf")