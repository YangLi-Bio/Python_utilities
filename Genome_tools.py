#################################################################################
#                                                                               #
# Functions to analyze transcriptomics data with the following characteristics: #
# 1. Large scale                                                                #
# 2. To be streamed with UCSC CellBrowser                                       #
# 3. To be streamed with deeplearning models                                    #
#                                                                               #
#################################################################################


# 1. calc_profile : calculate accessibility profiles based on genomic regions and 
#    BigWig files


script_dir = "/fs/ess/PCON0022/liyang/Python_utilities/Functions"



#################################################################################
#                                                                               #
# 1. calc_profile : calculate accessibility profiles based on genomic regions   #
#    and BigWig files                                                           #
#                                                                               #
#################################################################################


# Input :
# 1. regions : a list of regions
# 2. bw_files : a list of paths of BigWig files
# 3. width : the diameter of genomic interval
# 4. type : type of values to extract from each bin
# 5. nBins : 


def calc_profile(region_file, bw_files, width = 750, val_type = "max", nBins = 50):
    
    # Import modules
    import pyBigWig
    import os
    import numpy
    
    
    # Load regions file
    with open(region_file) as f:
        regions = f.readlines()
    print ("Finished loading " + str(len(regions)) + " genomics regions.")
    
    
    # Load files
    bw_list = list()
    for i in bw_files:
        bw_file = i
        
        # Calculate max value of the bigwig within our region
        print ("Reading the BigWig file via path: " + bw_file + " ...")
        bw = pyBigWig.open(bw_file)
        bw_list.append(bw)
    print ("Loaded " + str(len(bw_list)) + " BigWig files.")
    
    
    # Compute summary information on a range
#    arr_list = list()
    for i in range((len(bw_list))):
        bin_arr = []
        print ("Processing file: " + bw_files[i], " ...")
        for j in regions:
            ln_str = j.split("-")
            mid = round((int(ln_str[1]) + int(ln_str[2])) / 2)
            begin = mid - width
            end = mid + width
            prof = bw_list[i].stats(ln_str[0], begin, end, type = val_type, nBins = nBins)
            bin_arr.append(prof)
        numpy.savetxt(bw_files[i].replace("bigWig", "prof"), bin_arr, delimiter = ",")
#        arr_list.append(bin_arr)
    print ("Finished calculating binned profiles for " + str(len(regions)) + 
    " regions and " + str(len(bw_files)) + " BigWig files.")
