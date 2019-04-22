# IFT6135-Assignment3         
Team :         
Jean-Philippe Gagnon Fleury        
Hugo Côté        
Ahmadreza Godarzvandchegini      
Srinivas Venkattaramanujam      

Repo for IFT6135 Assignment3       
Data from svhn are not directly included, but the notebook should load them

sample_directory contains samples from GAN and VAE       
from_TA contains the code given by the TA's          
GAN_save contains state_dict for the GAN          
GAN_save/samples contains samples from the GAN during training          
GAN_save/samples/from_rez contains 2 trained GAN, one with make_GAN_decoder(100,64), one with make_GAN_decoder(100,128)           
old/save_5  contains our best VAE

Q1   : code for part 1     
density_estimation : code for part 1         
A3P2 : code for part 2          
Part3: code for part 3 (training + Qualitative evaluation) for VAE             
Part3_GAN : code for part 3 (training + Qualitative evaluation) for GAN                

from_TA\score_fid -> contains implementation of calculate_fid_score            
from its directory use the following command line to obtain FID score for the models :            
python score_fid.py ../sample_directory/VAE         
python score_fid.py ../sample_directory/GAN128             


