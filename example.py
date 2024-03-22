from util_funcs import  *
id=['NM_024090.3','NM_001286708.2','NM_001354158.2','NM_001318509.2','NM_001330174.3',
            'NM_000019.4']
filename='example.fasta'
download_genbank_sequences(email='1213@qq.com', accession_ids=id, output_filename=filename)
seq=find_sequence(fasta_file='example.fasta')
seq0=['CCCTCGTCGTCGTATCCC']*1
seq=seq0+seq+seq0
process_sequences_Long(seq=seq,species='human',folder_name='csv_slices_415nt')
process_sequences(seq=seq,species='human',folder_name='csv_slices_21nt',exact=True)
scores_Long=Long_scores_computing(seq=seq,species='human')
scores_short=scores_computing(seq=seq,species='human')
save_list_to_txt(scores_Long,'scores_Long.txt')
save_list_to_txt(scores_short,'scores_short.txt')
prediction_Long(seq,species='human',folder_name='csv_trial1')
prediction_short(seq,species='human',folder_name='csv_trial2')
prediction_Long_weighted(seq,species='human',folder_name='csv_trial3',pdf=True)
prediction_short_weighted(seq,species='human',folder_name='csv_tria4',pdf=True)
input_filename = "example.fasta"
output_filename = "example.csv"
fasta_to_csv(input_filename, output_filename)
############
seq1=['CCTATTAATGCTGAGAGATCC','TTAAGGATTCCGTCCCTTACA','TTCGAAGCCCCGCCGGTCCGC']
seq_hm_long=['AGGCGCAGAGGAGGGCGGTGTTGAGACCGGCGGAGCGGCGGGACCCCTAGGTGGCGGAGGGACGCTCCGGGAAAGCGAGGGGCGCTACGAGCTCTGGCCCACGTGACCTGCCGGGGGCGGGAGCAGGGGGCGCGCCGGCCTCCTGCGGTGCCCCTGCCTTGGGGAGGGGCCGTGACCACCCGTCTGTCGCCCGAGGCGGCCGCCGCTGCACCTTCACCGCGTACCCGGGAC']
prediction_short_weighted(seq1,species='human',folder_name='csv_trial1_human',pdf=True)
calculate_pro(seq1,species='human',folder_name='csv_trial1/pro')
prediction_short_weighted(seq1,species='yeast',folder_name='csv_trial1_yeast',pdf=True)
calculate_pro(seq1,species='yeast',folder_name='csv_trial1_yeast/pro')
prediction_short_weighted(seq1,species='archaea',folder_name='csv_trial1_archaea',pdf=True)
calculate_pro(seq1,species='archaea',folder_name='csv_trial1_archaea')
prediction_Long_weighted(seq_hm_long,species='human',folder_name='csv_trial_long',pdf=True)
calculate_pro_long(seq_hm_long,species='human',folder_name='csv_trial_long/pro')