# FSPEN 
Refer to FSPEN: AN ULTRA-LIGHTWEIGHT NETWORK FOR REAL TIME SPEECH ENAHNCMENT.

Note that thera are some parameters setting mistakes in the original paper, so we modify some parameters to make model running succeed, for example:
1. the number of sub-bands in groups is set to {8,7,6,7,6}; 
2. the linear op in feature Merge is set to (66,32);
3. the linear op in feature split is set to (32, 66);