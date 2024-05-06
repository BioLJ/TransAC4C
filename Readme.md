# The transAC4C
The transAC4C is a deep learning architecture for prediction RNA sequences undergoing
the N4-acetylcytidine (ac4C) modification

Reference: Ruijie Liu, Yuanpeng Zhang, Qi Wang, Xiaoping Zhang, TransAC4C—a novel interpretable architecture for multi-species identification of N4-acetylcytidine sites in RNA with single-base resolution, Briefings in Bioinformatics, Volume 25, Issue 3, May 2024, bbae200, https://doi.org/10.1093/bib/bbae200
## What is the transAC4C?
- The transAC4C consistes of two models transAC4C-21nt and transAC4C-415nt trained in 
the data with single-base resolution or not respectively. 
- Both models have a common architecture: one Transformer encoder, one Bi-LSTM and four 
Conv1d layers.

## What can the transAC4C do?
- predicting ac4C sites with base-levle resolution through transAC4C-21nt
- predicting the possibility of sequence undergoing ac4C <= 415nt through transAC4C-415nt
- predicting the degree of the sequence undergoing ac4C of any length, which make
the comparison between different sequences possible
- predicting ac4C sites in multi species
- interpret the prediction by two methods (Attention weights and In-silico-mutagenesis)

## How to use transAC4C
__OS dependency: It has been tested successfully in Linux and MacOS__

__1.We believe you can use it easilly by directly downloading it from this repo or by__
```sh
git clone https://github.com/BioLJ/TransAC4C.git
```
if git is abled in your systmem

__2.Installing requiring packages:__

New environment  (can be skipped): 
if you use conda then it's suggested to build a new conda environment by
```sh
conda create --name the_name_of_your_environ python=3.8.10
source activate the_name_of_your_environ
```
Install packages (it takes some time)
```sh
cd TransAC4C ### go to the location of the file downloaded form this repo
pip install -r requirements.txt ### install the required packages
```
__3.Analyze sequences by transAC4C__

We prepared an example file and example python script, you can use it by changeing the corresponding parameter in python

The detailed functions are:

3.1 Files preparation
```sh
###### You can download as:
download_genbank_sequences(email=maiil, accession_ids=id, output_filename=filename)
###### emial: your email; accession_ids:the list accession id of genes; 
###### output_filename: name of your fasta file
```
3.2 Read fasta
```sh
###### read thef fasta you have prepared or downloaded
seq=find_sequence(fasta_file='example.fasta')
###### fasta_file: name of your fasta file
###### 
```
3.3 Analyze file
3.3.1 predicting ac4C for every sequences in every slices 
```sh
### sequences are sliced into 415nt, analyzed by transAC4C-415nt
process_sequences_Long(seq=seq,species='human',folder_name='csv_slices_415nt')
### seq:sequence; species: model of species: only human; folder_name:the folder of your result

####sequences are sliced into 21nt or less with cytidine in the middle, analyzed ####by transAC4C-21nt
process_sequences(seq=seq,species='human',folder_name='csv_slices_21nt',exact=True)
### seq:sequence; species: model of species: human, yeast, or archaea
###folder_name:the folder of your result, exact: True means only slices at 21nt 
###would be analyzed
```
3.3.2 caculating the ac4C score for every sequences 
```sh
## ac4C scores caculated by transAC4C-415nt or transAC4C-21nt
scores_Long=Long_scores_computing(seq=seq,species='human')
##### species:only human

scores_short=scores_computing(seq=seq,species='human')
##### species:human, yeast, or archaea
### save scores to txt
save_list_to_txt(scores_Long,'scores_Long.txt')
save_list_to_txt(scores_short,'scores_short.txt')
```
3.3.3 predicting whether the sequences undergoing ac4C for sequences within 415nt or 21nt
``` 
## As the sequences  within 415nt or 21nt are not very common, 
## therefore we mannually generate two sequences to go the following analysis
seq0=['CCCTCGTCGTCGTATCCC']*1
seq=seq0+seq+seq0
###
## only the sequences within 415nt was predicted 
prediction_Long(seq,species='human',folder_name='csv_trial1')
##### species:only human
### only the sequences within 21nt was predicted 
prediction_short(seq,species='human',folder_name='csv_trial2')
##### species:human, yeast, or archaea
```
3.3.4 predicting whether the sequences undergoing ac4C for sequences within 415nt or 21nt
with the interpretation by Attention Weights (Computational!!!)
``` 
## only the sequences within 415nt was predicted 
prediction_Long_weighted(seq,species='human',folder_name='csv_trial3',pdf=True)
##### species:only human; pdf=True means visiulized it 
### only the sequences within 21nt was predicted 
prediction_short_weighted(seq,species='human',folder_name='csv_tria4',pdf=True)
##### species:human, yeast, or archaea; pdf=True means visiulized it 
```
3.3.5 Interpretate the results by in-silico-mutagenesis (Computational!!!)
``` 
# interpretate the results of sequences <= 21nt by model-TransAC4C-21nt
calculate_pro(seq1,species='human',folder_name='csv_trial1_human/pro')
calculate_pro(seq1,species='yeast',folder_name='csv_trial1_yeast/pro')
calculate_pro(seq1,species='archaea',folder_name='csv_trial1_archaea/pro')
# interpretate the results of sequences <= 415nt by model-TransAC4C-415nt
calculate_pro_long(seq_hm_long,species='human',folder_name='csv_trial_long/pro')
``` 
3.3.6 Additional Function
make the fasta into a csv, easier for you to search it 
``` 
input_filename = "example.fasta"
output_filename = "example.csv"
fasta_to_csv(input_filename, output_filename)
``` 
__Note again: the aforementioned analysis should be conducted in the 'TransAC4C' dir
