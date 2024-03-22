import re
import sys
import pandas as pd
import numpy as np
import torch,gc
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from Bio import SeqIO
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from util_datasets import *
import itertools
import os
from Bio import Entrez
import matplotlib.pyplot as plt
from matplotlib import rcParams
##
rcParams['font.size'] = 14
##
def find_sequence(fasta_file):
    def extract_alpha_chars(input_string):
        alpha_chars = ''.join(filter(str.isalpha, input_string))
        return alpha_chars
    fasta_seq = [extract_alpha_chars(fa.seq) for fa in SeqIO.parse(fasta_file, "fasta")]
    max_length =300
    fasta_seq = fasta_seq
    seq = []
    for i in range(len(fasta_seq)):
        seq.append(str(fasta_seq[i]))
    assert len(seq) == len(fasta_seq)
    return seq
#
def generate_indices(sequence, char_dict, desired_length=413):
    indices = []
    for i in range(len(sequence) - len(next(iter(char_dict)))+1):
        triplet = sequence[i:i+len(next(iter(char_dict)))]
        if triplet in char_dict:
            idxs = char_dict[triplet]
            indices.extend(idxs)
    while len(indices) < desired_length:
        indices.append(0)

    return indices
#
def generate_char(letters, length):
    combinations = itertools.product(letters, repeat=length)
    char_dict = {}
    for idx, combo in enumerate(combinations, start=1):
        char = ''.join(combo)  # 将元组转换为字符串
        if char not in char_dict:
            char_dict[char] = []
        char_dict[char].append(char)

    return char_dict
###
def generate_char_dict(letters, length):
    combinations = itertools.product(letters, repeat=length)
    char_dict = {}
    for idx, combo in enumerate(combinations, start=1):
        char = ''.join(combo)  
        if char not in char_dict:
            char_dict[char] = []
        char_dict[char].append(idx)

    return char_dict
####
def save_list_to_txt(my_list, file_name):
    with open(file_name, 'w') as file:
        for item in my_list:
            file.write("%s\n" % item)
def is_sequence(sequence):
    # 使用正则表达式检查字符串是否只包含AGCT构成的字符
    pattern = re.compile(r'^[AGCT]+$')
    if pattern.match(sequence):
        print("Analysis begin")
    else:
        print("Error:containg no AGCT, analysis break")
        sys.exit()

def generate_subsequences_with_c(sequence, length=21, exact=True):
    subsequences = []
    c_positions = []
    
    for i in range(len(sequence)):
        if sequence[i] == 'C':
            start = max(0, i - length // 2)
            end = min(len(sequence), i + (length + 1) // 2)
            subsequence = None  
            if exact:
                #print(end,i,start)
                if end - i-1 == i - start == 10:
                    subsequence = sequence[start:end]
            else:
                if end - i-1 == i - start:
                    subsequence = sequence[start:end]
                elif i - start < end - i-1:
                    subsequence = sequence[start:2 * i - start + 1]
                elif i - start > end - i-1:
                    subsequence = sequence[2 * i - end + 1:end]
            if subsequence is not None:
                c_positions.append(i + 1)
                subsequences.append(subsequence)   
    return c_positions, subsequences
#####
def slice_elements(input_list):
    N = len(input_list)
    slice_size = 415
    slices = []
    if N<=415:
        slices.append(input_list)
    else:
        for i in range(0, N-414):
            slice_end = i + slice_size if i + slice_size <= N else N
            sliced_element = input_list[i:slice_end]
            slices.append(sliced_element)
    return slices
###
def predict_no_weights(model,x_dataloader,device):
    prediction=[]
    probalility=[]
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(x_dataloader):
             data=data.long().to(device).clone().detach()
             test_pred,w = model(data.to(device))
             probabilities = nn.functional.softmax(test_pred , dim=1)
             _, test_pred  = probabilities.max(1) 
             if test_pred.size()[0]==1:
                probalility += [probabilities[:,1].detach().cpu().flatten().squeeze().tolist()]
                test_label = test_pred.detach().cpu().numpy().flatten()
                prediction += [test_label.squeeze().tolist()]
                #print(probabilities[:,1].detach().cpu().flatten().squeeze().tolist())
             else:
                probalility += probabilities[:,1].detach().cpu().flatten().squeeze().tolist()
                test_label = test_pred.detach().cpu().numpy().flatten()
                prediction += test_label.squeeze().tolist()
    return prediction,probalility 
###
def predict(model,x_dataloader,device):
    prediction=[]
    weights=[0]
    probalility=[]
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(x_dataloader):
             data=data.long().to(device).clone().detach()
             test_pred,w = model(data.to(device))
             probabilities = nn.functional.softmax(test_pred , dim=1)
             _, test_pred  = probabilities.max(1) 
             if test_pred.size()[0]==1:
                probalility += [probabilities[:,1].detach().cpu().flatten().squeeze().tolist()]
                test_label = test_pred.detach().cpu().numpy().flatten()
                prediction += [test_label.squeeze().tolist()]
                #print(probabilities[:,1].detach().cpu().flatten().squeeze().tolist())
             else:
                probalility += probabilities[:,1].detach().cpu().flatten().squeeze().tolist()
                test_label = test_pred.detach().cpu().numpy().flatten()
                prediction += test_label.squeeze().tolist()
             for i in range(len(w)):
                 w[i]=w[i].detach().cpu().numpy()
                 w[i] = w[i].squeeze()
             w=np.array(w).squeeze()
             if isinstance(weights, list):
                 weights=w
                 weights=weights.squeeze()
             else:
                 weights=np.concatenate((weights, w),axis=0)

             #print(weights.shape)
    return prediction,probalility,weights

#####
def process_sequences(seq,species,folder_name='csv_trial',exact=False):
    if species =='human':
        model_path = 'model/transAC4C-21nt/hm'
    elif species =='archaea':
        model_path = 'model/transAC4C-21nt/archaea'
    elif species == 'yeast':
        model_path = 'model/transAC4C-21nt/yeast'
    else:
        raise ValueError("Invalid species. Accepted values are 'human', 'archaea', or 'yeast'.")
    directory=model_path
    for i, sequence in enumerate(seq):
        results = []
        os.makedirs(folder_name,exist_ok=True)
        is_sequence(sequence)
        c_positions, sequence = generate_subsequences_with_c(sequence,exact=exact)
        letters = ['A', 'T', 'C', 'G']
        char_dict = generate_char_dict(letters, length=3)
        indices = [generate_indices(seq, char_dict, desired_length=19) for seq in sequence]
        sequence_tensor = torch.from_numpy(np.array(indices))

        # set
        test_set = MyDataset(sequence_tensor, None)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)
        # prediction
        all_files = os.listdir(directory)
        ckpt_files = [filename for filename in all_files if filename.endswith('.pth')]
        probalilitys=[]
        for filename in ckpt_files:
            filename=directory+'/'+filename
            model = torch.load(filename)
            prediction,probalility =predict_no_weights(model=model
                  ,x_dataloader=test_loader,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            probalilitys.append(probalility)
        probalilitys_array = np.array(probalilitys)
        average_probabilities = np.mean(probalilitys_array, axis=0)
        prediction = (average_probabilities >= 0.5).astype(int)
        probability=average_probabilities
        npp=np.array(probability) 
        scores=np.mean(npp)

        # result to df
        df=pd.DataFrame({
            "c_positions": c_positions,
            "sequence": sequence,
            "prediction": prediction,
            "probability": probability,
            'Scores for this whole sequence':scores
        })
        name=folder_name+'/'+'output'+str(i+1)+'.csv'
        df.to_csv(name,index = None,encoding = 'utf8')
#####
def scores_computing(seq,species,exact=True):
    all_scores=[]
    if species =='human':
        model_path = 'model/transAC4C-21nt/hm'
    elif species =='archaea':
        model_path = 'model/transAC4C-21nt/archaea'
    elif species == 'yeast':
        model_path = 'model/transAC4C-21nt/yeast'
    else:
        raise ValueError("Invalid species. Accepted values are 'human', 'archaea', or 'yeast'.")
    directory=model_path
    for i, sequence in enumerate(seq):
        results = []
        is_sequence(sequence)
        c_positions, sequence = generate_subsequences_with_c(sequence,exact=exact)
        letters = ['A', 'T', 'C', 'G']
        char_dict = generate_char_dict(letters, length=3)
        indices = [generate_indices(seq, char_dict, desired_length=19) for seq in sequence]
        sequence_tensor = torch.from_numpy(np.array(indices))

        # 构建测试数据集和数据加载器
        test_set = MyDataset(sequence_tensor, None)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)
        ########################################################################
        # 进行预测
        all_files = os.listdir(directory)
        ckpt_files = [filename for filename in all_files if filename.endswith('.pth')]
        probalilitys=[]
        for filename in ckpt_files:
            filename=directory+'/'+filename
            #print(filename)
            model = torch.load(filename)
            prediction,probalility =predict_no_weights(model=model
                  ,x_dataloader=test_loader,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            probalilitys.append(probalility)
        probalilitys_array = np.array(probalilitys)
        average_probabilities = np.mean(probalilitys_array, axis=0)
        #print(average_probabilities.shape)
        prediction = (average_probabilities >= 0.5).astype(int)
        probability=average_probabilities
        #npp=np.array([p for p in probability  if p > 0.5]) 
        #npp=np.array(probability )
        npp=np.array(probability)
        if len(npp) == 0:
            npp = np.array([0])
        scores=np.sum(npp)/len(npp)
        all_scores.append((scores))
    return all_scores
#################
def process_sequences_Long(seq,species='human',folder_name='csv_trial'):

    if species =='human':
        model_path = 'model/transAC4C-415nt/model.pth'
    else:
        raise ValueError("Invalid species. Accepted values is 'human")
    model = torch.load(model_path)
    for i, sequence in enumerate(seq):
        os.makedirs(folder_name,exist_ok=True)
        is_sequence(sequence)
        sequence = slice_elements(sequence)
        letters = ['A', 'T', 'C', 'G']
        char_dict = generate_char_dict(letters, length=3)
        indices = [generate_indices(seq, char_dict, desired_length=413) for seq in sequence]
        sequence_tensor = torch.from_numpy(np.array(indices))
        # 构建测试数据集和数据加载器
        test_set = MyDataset(sequence_tensor, None)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)


        # 进行预测
        prediction, probability = predict_no_weights(model=model, 
                                                     x_dataloader=test_loader,
                                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        npp=np.array(probability) 
        scores=np.mean(npp)

        # 将结果保存到列表
        df=pd.DataFrame({
            "sequence": sequence,
            "prediction": prediction,
            "probability": probability,
            'Scores for this whole sequence':scores
        })
        # 将列表转换为DataFrame
        #df = pd.DataFrame(results)
        # 将DataFrame写入CSV文件
        name=folder_name+'/'+'output'+str(i+1)+'.csv'
        df.to_csv(name,index = None,encoding = 'utf8')
##
def Long_scores_computing(seq,species='human'):
    all_scores=[]
    if species =='human':
        model_path = 'model/transAC4C-415nt/model.pth'
    else:
        raise ValueError("Invalid species. Accepted values is 'human")
    model = torch.load(model_path)
    for i, sequence in enumerate(seq):
        results = []
        is_sequence(sequence)
        sequence = slice_elements(sequence)
        letters = ['A', 'T', 'C', 'G']
        char_dict = generate_char_dict(letters, length=3)
        indices = [generate_indices(seq, char_dict, desired_length=413) for seq in sequence]
        sequence_tensor = torch.from_numpy(np.array(indices))
        # 构建测试数据集和数据加载器
        test_set = MyDataset(sequence_tensor, None)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)

        # 进行预测
        prediction, probability = predict_no_weights(model=model, 
                                                     x_dataloader=test_loader,
                                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #npp=np.array([p for p in probability  if p > 0.5]) 
        #print(npp)
        npp=np.array(probability )
        if len(npp) == 0:
            npp = np.array([0])
        #print(npp)
        scores=np.sum(npp)/len(npp)
        #print(scores)
        all_scores.append(scores)
    return all_scores
###### 
def prediction_Long(seq,species='human',folder_name='csv_trial'):
    os.makedirs(folder_name,exist_ok=True)
    results = []
    if species =='human':
        model_path = 'model/transAC4C-415nt/model.pth'
    else:
        raise ValueError("Invalid species. Accepted values is 'human")
    model = torch.load(model_path)

    for i, sequence in enumerate(seq):
        if len(sequence)<=415:
            is_sequence(sequence)
            sequence = slice_elements(sequence)
            letters = ['A', 'T', 'C', 'G']
            #equence=[sequence]
            char_dict = generate_char_dict(letters, length=3)
            indices = [generate_indices(seq, char_dict, desired_length=413) for seq in sequence]
            sequence_tensor = torch.from_numpy(np.array(indices))
            test_set = MyDataset(sequence_tensor, None)
            test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)


            prediction, probability = predict_no_weights(model=model, 
                                                     x_dataloader=test_loader,
                                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            results.append({
            "Name": f"sequence_{i+1}",
            "sequence": sequence,
            "prediction": prediction,
            "probability": probability
            })

    # 将列表转换为DataFrame
    #print(results)
    df = pd.DataFrame(results)
    name=folder_name+'/'+'prediction_result'+'.csv'
    df.to_csv(name,index = None,encoding = 'utf8')
    print(name+' has been created')
####
def prediction_short(seq,species='human',folder_name='csv_trial'):
    os.makedirs(folder_name,exist_ok=True)
    results = []
    if species =='human':
        model_path = 'model/transAC4C-21nt/hm'
    elif species =='archaea':
        model_path = 'model/transAC4C-21nt/archaea'
    elif species == 'yeast':
        model_path = 'model/transAC4C-21nt/yeast'
    else:
        raise ValueError("Invalid species. Accepted values are 'human', 'archaea', or 'yeast'.")
    directory=model_path
    for i, sequence in enumerate(seq):
        if len(sequence)<=21:
            is_sequence(sequence)
            sequence = [sequence]
            letters = ['A', 'T', 'C', 'G']
            char_dict = generate_char_dict(letters, length=3)
            indices = [generate_indices(seq, char_dict, desired_length=19) for seq in sequence]
            sequence_tensor = torch.from_numpy(np.array(indices))
            test_set = MyDataset(sequence_tensor, None)
            test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)


            all_files = os.listdir(directory)
            ckpt_files = [filename for filename in all_files if filename.endswith('.pth')]
            probalilitys=[]
            for filename in ckpt_files:
                filename=directory+'/'+filename
                #print(filename)
                model = torch.load(filename)
                prediction,probalility =predict_no_weights(model=model
                  ,x_dataloader=test_loader,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                probalilitys.append(probalility)
            probalilitys_array = np.array(probalilitys)
            average_probabilities = np.mean(probalilitys_array, axis=0)
            #print(average_probabilities.shape)
            prediction = (average_probabilities >= 0.5).astype(int)
            probability= average_probabilities

            results.append({
            "Name": f"sequence_{i+1}",
            "sequence": sequence,
            "prediction": prediction,
            "probability": probability
            })
    df = pd.DataFrame(results)
    name=folder_name+'/'+'prediction_result'+'.csv'
    df.to_csv(name,index = None,encoding = 'utf8')
    print(name+' has been created')
###
def download_genbank_sequences(email, accession_ids, output_filename):
    Entrez.email = email
    
    with open(output_filename, "a") as output_file:
        for record_id in accession_ids:
            result_handle = Entrez.efetch(db="nucleotide", rettype="gb", id=record_id)
            seq_record = SeqIO.read(result_handle, format='gb')
            result_handle.close()
            output_file.write(seq_record.format('fasta'))
    
    print(f"FASTA '{output_filename}' generated!!!。")
###
def fasta_to_csv(input_filename, output_filename):
    sequences = []
    names = []
    
    # 读取FASTA文件
    with open(input_filename, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            names.append(record.id)
            sequences.append(str(record.seq))
    
    # 创建DataFrame
    df = pd.DataFrame({
        "Name": names,
        "Sequence": sequences
    })
    
    # 将DataFrame保存为CSV文件
    df.to_csv(output_filename, index=False)
    print(f"CSV '{output_filename}' generated!!!")
###
def plot_bar_charts(seq_print, weights, folder_name='all_png',number=0,x_axis=5,legend_name='Weights',base=False):
    os.makedirs(folder_name, exist_ok=True)
    def plot_bar_chart(seq, weights, folder_name, i,x_axis=5):
        if base==False:
            seq_length = len(seq)-2
        else:
            seq_length = len(seq)
        result_list = [i for i in range(1, seq_length + 1)]
        y = np.array(weights[:seq_length], dtype=np.float16)
        x = np.array(result_list, dtype=np.float16)

        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
        extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
        im = ax.imshow(y[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        x_ticks = range(0, len(x), x_axis)
        ax2.set_xticks(x_ticks)
        #ax2.set_xticklabels([str(int(x[i])) for i in x_ticks])
        ax2.plot(x, y)
        ax2.set_xlabel('Position starting from 1')
        ax2.set_ylabel('Importance')
        ax.set_xlabel('Position starting from 1')
        ax.set_ylabel('Importance')
        # Add colorbar
        cbar = fig.colorbar(im, ax=[ax, ax2], location='right', shrink=0.6, pad=0.05)
        cbar.set_label(legend_name)

        #plt.tight_layout()

        plt.savefig(f'{folder_name}/weights_colored_{i}.pdf')
        plt.close()
    plot_bar_chart(seq_print, weights, folder_name, number,x_axis)

#########################
def prediction_short_weighted(seq,species='human',folder_name='csv_trial',pdf=True):
    os.makedirs(folder_name,exist_ok=True)
    df_w=pd.DataFrame()
    results = []
    if species =='human':
        model_path = 'model/transAC4C-21nt/hm'
    elif species =='archaea':
        model_path = 'model/transAC4C-21nt/archaea'
    elif species == 'yeast':
        model_path = 'model/transAC4C-21nt/yeast'
    else:
        raise ValueError("Invalid species. Accepted values are 'human', 'archaea', or 'yeast'.")
    directory=model_path
    for i, sequence in enumerate(seq):
        if len(sequence)<=21:
            is_sequence(sequence)
            sequence = [sequence]
            letters = ['A', 'T', 'C', 'G']
            char_dict = generate_char_dict(letters, length=3)
            indices = [generate_indices(seq, char_dict, desired_length=19) for seq in sequence]
            sequence_tensor = torch.from_numpy(np.array(indices))
            test_set = MyDataset(sequence_tensor, None)
            test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)


            all_files = os.listdir(directory)
            ckpt_files = [filename for filename in all_files if filename.endswith('.pth')]
            probalilitys=[]
            weight=[]
            for filename in ckpt_files:
                filename=directory+'/'+filename
                #print(filename)
                model = torch.load(filename)
                prediction,probalility,weights=predict(model=model
                  ,x_dataloader=test_loader,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                probalilitys.append(probalility)
                #print(weights[0])
                weights=np.mean(weights,axis=0)
                #print(weights[0])
                weight.append(weights)
            #print(np.array(weight).shape)
            #weight=np.array(weight).transpose(1, 0, 2)
            weight = np.mean(weight, axis=0)
            weights=weight
            #print(weights)
            probalilitys_array = np.array(probalilitys)
            average_probabilities = np.mean(probalilitys_array, axis=0)
            #print(average_probabilities.shape)
            prediction = (average_probabilities >= 0.5).astype(int)
            probability= average_probabilities
            #weights=np.mean(weights,axis=0)
            sequence_df = pd.DataFrame(weights)
            sequence_df = sequence_df.T
            df_w = pd.concat([df_w, sequence_df])

            results.append({
            "Name": f"sequence_{i+1}",
            "sequence": sequence,
            "prediction": prediction,
            "probability": probability
            })
            if pdf==True:
                plot_bar_charts(seq_print=sequence[0],weights=weights,folder_name=(folder_name+'/weights_pdf'),number=i+1,
                            x_axis=5)
    df = pd.DataFrame(results)
    name=folder_name+'/'+'outputs'+'.csv'
    df.to_csv(name,index = None,encoding = 'utf8')
    print(name+' has been created')
    ####
    name=folder_name+'/'+'weights'+'.csv'
    df_w.to_csv(name,index = None,encoding = 'utf8')
    print(name+' has been created')
#################
def prediction_Long_weighted(seq,species='human',folder_name='csv_trial',pdf=True):
    os.makedirs(folder_name,exist_ok=True)
    df_w=pd.DataFrame()
    results = []
    if species =='human':
        model_path = 'model/transAC4C-415nt/model.pth'
    else:
        raise ValueError("Invalid species. Accepted values is 'human")
    model = torch.load(model_path)
    for i, sequence in enumerate(seq):
        if len(sequence)<=415:
            is_sequence(sequence)
            sequence = [sequence]
            letters = ['A', 'T', 'C', 'G']
            char_dict = generate_char_dict(letters, length=3)
            indices = [generate_indices(seq, char_dict, desired_length=413) for seq in sequence]
            sequence_tensor = torch.from_numpy(np.array(indices))
            test_set = MyDataset(sequence_tensor, None)
            test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)


            prediction, probability,weights= predict(model=model, 
                                                     x_dataloader=test_loader,
                                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            weights=np.mean(weights,axis=0)
            sequence_df = pd.DataFrame(weights)
            sequence_df = sequence_df.T
            df_w = pd.concat([df_w, sequence_df])

            results.append({
            "Name": f"sequence_{i+1}",
            "sequence": sequence,
            "prediction": prediction,
            "probability": probability
            })
            if pdf==True:
                plot_bar_charts(seq_print=sequence[0],weights=weights,folder_name=(folder_name+'/weights_pdf'),number=i+1,
                            x_axis=25)
    df = pd.DataFrame(results)
    name=folder_name+'/'+'outputs'+'.csv'
    df.to_csv(name,index = None,encoding = 'utf8')
    print(name+' has been created')
    ####
    name=folder_name+'/'+'weights'+'.csv'
    df_w.to_csv(name,index = None,encoding = 'utf8')
    print(name+' has been created')
####
def calculate_pro(sequences, species,folder_name):
    pro_list = []
    def short(seq):
        average_probabilities_t=[]
        for i, sequence in enumerate(seq):
            if len(sequence)<=21:
                is_sequence(sequence)
                sequence = [sequence]
                letters = ['A', 'T', 'C', 'G']
                char_dict = generate_char_dict(letters, length=3)
                indices = [generate_indices(seq, char_dict, desired_length=19) for seq in sequence]
                sequence_tensor = torch.from_numpy(np.array(indices))
                test_set = MyDataset(sequence_tensor, None)
                test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)

                all_files = os.listdir(directory)
                ckpt_files = [filename for filename in all_files if filename.endswith('.pth')]
                probalilitys=[]
                for filename in ckpt_files:
                    filename=directory+'/'+filename
                    model = torch.load(filename)
                    prediction,probalility =predict_no_weights(model=model
                      ,x_dataloader=test_loader,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    probalilitys.append(probalility)
                probalilitys_array = np.array(probalilitys)
                average_probabilities = np.mean(probalilitys_array, axis=0)
                probability= average_probabilities
                average_probabilities_t.append(probability)
        return average_probabilities_t 

    if species =='human':
        model_path = 'model/transAC4C-21nt/hm'
    elif species =='archaea':
        model_path = 'model/transAC4C-21nt/archaea'
    elif species == 'yeast':
        model_path = 'model/transAC4C-21nt/yeast'
    else:
        raise ValueError("Invalid species. Accepted values are 'human', 'archaea', or 'yeast'.")
    directory=model_path
    average_probabilities_t = short(sequences)
    nun_sequence=0
    for i in range(len(sequences)):
        validneg_circle=sequences[i]
        if len(validneg_circle)<=21:
            new_list = []
            for k in range(len(validneg_circle)):
                for base in ['A', 'C', 'G', 'T']:  # 遍历A、C、G、T
                    if validneg_circle[k] != base:
                        new_sequence = validneg_circle[:k] + base + validneg_circle[k+1:]
                        new_list.append(new_sequence)
            indicess=[]
            letters = ['A', 'T', 'C', 'G']
            char_dict = generate_char_dict(letters, length=3)
            for sequence in new_list:
                indice = generate_indices(sequence, char_dict, desired_length=19)
                indicess.append(indice)
            seque = torch.from_numpy(np.array(indicess))
            test_set = MyDataset(seque, None)
            test_loader = DataLoader(test_set, batch_size=256, shuffle=False, pin_memory=True)
            all_files = os.listdir(directory)
            ckpt_files = [filename for filename in all_files if filename.endswith('.pth')]
            probalilitys=[]
            for filename in ckpt_files:
                filename=directory+'/'+filename
                model = torch.load(filename)
                _,probalility=predict_no_weights(model=model
                      ,x_dataloader=test_loader,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                probalilitys.append(probalility)
            probalilitys_array = np.array(probalilitys)
            average_probabilities = np.mean(probalilitys_array, axis=0).flatten()
            reshaped_array = average_probabilities.reshape(-1, 3)
            min_values = np.min(reshaped_array, axis=1).flatten()
            #print(i,average_probabilities_t)
            pro=-(min_values-average_probabilities_t[nun_sequence])
            plot_bar_charts(seq_print=validneg_circle,weights=pro,folder_name=(folder_name+'/in_silico_mutat_pdf'),number=i+1,
                            x_axis=5,legend_name='Changes of Probability',base=True)
            pro_list.append(pro)
            nun_sequence=nun_sequence+1
    df_data = []
    for i, pro_values in enumerate(pro_list, start=0):
        row = {"sequence": f"sequence_{i}", **{f"Position_{j+1}": value for j, value in enumerate(pro_values, start=0)}}
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(f"{folder_name}/in_silico_mutagenesis.csv", index=False)
#########
def calculate_pro_long(sequences, species,folder_name):
    pro_list = []
    def long(seq):
        average_probabilities_t=[]
        for i, sequence in enumerate(seq):
            if len(sequence)<=415:
                is_sequence(sequence)
                sequence = [sequence]
                letters = ['A', 'T', 'C', 'G']
                char_dict = generate_char_dict(letters, length=3)
                indices = [generate_indices(seq, char_dict, desired_length=413) for seq in sequence]
                sequence_tensor = torch.from_numpy(np.array(indices))
                test_set = MyDataset(sequence_tensor, None)
                test_loader = DataLoader(test_set, batch_size=8, shuffle=False, pin_memory=True)
                probalilitys=[]
                prediction,probalility =predict_no_weights(model=model
                      ,x_dataloader=test_loader,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                probalilitys.append(probalility)
                probalilitys_array = np.array(probalilitys)
                probability= probalilitys_array 
                average_probabilities_t.append(probability)
        return average_probabilities_t 

    if species =='human':
        model_path = 'model/transAC4C-415nt/model.pth'
    else:
        raise ValueError("Invalid species. Accepted values is 'human")
    model = torch.load(model_path)
    average_probabilities_t = long (sequences)
    for i in range(len(sequences)):
        validneg_circle=sequences[i]
        #print(sequences)
        new_list = []
        for k in range(len(validneg_circle)):
            for base in ['A', 'C', 'G', 'T']:  # 遍历A、C、G、T
                if validneg_circle[k] != base:
                    new_sequence = validneg_circle[:k] + base + validneg_circle[k+1:]
                    new_list.append(new_sequence)
        indicess=[]
        #print(new_list)
        letters = ['A', 'T', 'C', 'G']
        char_dict = generate_char_dict(letters, length=3)
        for sequence in new_list:
            indice = generate_indices(sequence, char_dict, desired_length=413)
            indicess.append(indice)
        seque = torch.from_numpy(np.array(indicess))
        test_set = MyDataset(seque, None)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, pin_memory=True)
        _,probalility=predict_no_weights(model=model
                      ,x_dataloader=test_loader,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        probalilitys_array = np.array(probalility)
        average_probabilities = probalilitys_array.flatten()
        reshaped_array = average_probabilities.reshape(-1, 3)
        min_values = np.min(reshaped_array, axis=1).flatten()
        average_probabilities_t_expanded = np.full_like(min_values, average_probabilities_t[i])
        pro = -(min_values - average_probabilities_t_expanded)
        #print(pro.shape)
        plot_bar_charts(seq_print=validneg_circle,weights=pro,folder_name=(folder_name+'/in_silico_mutat_pdf'),number=i+1,
                            x_axis=50,legend_name='Changes of Probability',base=True)
        pro_list.append(pro)
    df_data = []
    for i, pro_values in enumerate(pro_list, start=0):
        row = {"sequence": f"sequence_{i}", **{f"Position_{j+1}": value for j, value in enumerate(pro_values, start=0)}}
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(f"{folder_name}/in_silico_mutagenesis.csv", index=False)