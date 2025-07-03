import torch
import pandas as pd
import numpy as np
import tifffile
import anndata as ad
import scanpy as sc
from model.NPF import NPF
from utils.tools import *
from utils.dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='seed',default=42)
parser.add_argument('--Dateset', type=str, help='Dateset name',default='10xVisium')
parser.add_argument('--sample', type=str, help='sample name',default='CytAssist_FFPE_Protein_Expression_Human_Tonsil')
parser.add_argument('--output',type=str,default="../results/NPF",help='output folder path')
parser.add_argument('--trainset',type=int,default=None,help='only for Pseudo-Visium Dateset')
cfg = parser.parse_args()

if __name__ == "__main__":
    logdir = os.path.join(cfg.output,f'Dateset_{cfg.Dateset}',f'{cfg.sample}',f'seed_{cfg.seed}')
    print("Log:", logdir)
    log = Log(logdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("use device:",device)

    if cfg.Dateset == '10xVisium':
        datadir = os.path.join('../data',cfg.Dateset,cfg.sample) 
        if os.path.exists(os.path.join(datadir,'tissue_image.tif')):
            he_tiffile = os.path.join(datadir,'tissue_image.tif')
        elif os.path.exists(os.path.join(datadir,'tissue_image.btf')):
            he_tiffile = os.path.join(datadir,'tissue_image.btf')
        he_tif = tifffile.imread(he_tiffile)
        ymax,xmax,_ = he_tif.shape
        print("HE image shape:",he_tif.shape)
        adata_file_path = os.path.join(datadir,'filtered_feature_bc_matrix.h5')
        tissue_positions = os.path.join(datadir,'spatial','tissue_positions.csv')
        adata = sc.read_10x_h5(adata_file_path,gex_only=False)
        positions = pd.read_csv(
            tissue_positions,
            header=0,
            index_col=0,
        )
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]
        adata.obs = adata.obs.join(positions, how="left")
        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )
        sc.pp.log1p(adata)
        adata = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()
    

    adata.X = adata.X.toarray().astype(np.float32)
    proteinname = adata.var['gene_ids'].to_list()
    train_val_adata,test_adata = split_adata_random(adata,0.8,xmax,ymax) 
    train_adata,valid_adata = split_adata_random(train_val_adata,7/8,xmax,ymax) 
    print(train_adata)
    print(valid_adata)
    print(test_adata)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = SampleDataset(train_adata,he_tif,224,transform=train_transform)
    valid_dataset = SampleDataset(valid_adata,he_tif,224,transform=valid_transform)
    test_dataset = SampleDataset(test_adata,he_tif,224,transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=4)

    model = NPF(num = train_adata.n_vars)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001,weight_decay=1e-5)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_start_lr=1e-6,warmup_epochs=5, max_epochs=100,eta_min = 1e-6)
    Loss_function = nn.MSELoss()

    bestvalpcc = 0
    bestvalmse = 100
    for epoch in range(100):
        current_lr = scheduler.get_last_lr()
        print(f"Current learning rate: {current_lr}")
        model.train()

        train_total_loss = 0
        prelist = []
        for i in range(len(proteinname)):
            prelist.append([])
        gtlist = []
        for i in range(len(proteinname)):
            gtlist.append([])

        print_it = 0
        for iteration, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs,pos, label = data
            pos = pos.to(device)
            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model(inputs,pos) 
            loss = Loss_function(outputs, label)
            
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
            log.append_iter_loss(loss.item())

            print_it += 1
            if print_it % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {print_it}, Loss: {loss.item()}")

        train_loss = train_total_loss/len(train_loader)
        print(f"epoch {epoch} train loss:",train_loss )
        
        scheduler.step()

        model.eval()
        valid_total_loss = 0
        for iteration, data in enumerate(valid_loader):
            inputs,pos, label = data
            pos = pos.to(device)
            inputs = inputs.to(device)
            label = label.to(device)
            with torch.no_grad():
                outputs = model(inputs,pos) 
                loss = Loss_function(outputs, label)
                valid_total_loss += loss.item()
                
        valid_loss = valid_total_loss/len(valid_loader)

        print(f"epoch {epoch} valid loss:",valid_loss )
        log.append_epoch_loss(train_loss,valid_loss)
        if bestvalmse > valid_loss:
            bestvalmse = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(log.save_path, "bestmodel.pth"),
            )
            print("save best model")

    print("load best model:")
    model.to('cpu')
    bestmodel_dict = torch.load(os.path.join(log.save_path, "bestmodel.pth"), map_location = 'cpu' )
    model.load_state_dict(bestmodel_dict)
    model.to('cuda')
    pmse_sum = np.zeros(len(proteinname))
    pmae_sum = np.zeros(len(proteinname))
    test_total_loss = 0
    prelist = []
    for i in range(len(proteinname)):
        prelist.append([])
    gtlist = []
    for i in range(len(proteinname)):
        gtlist.append([])
    for iteration, data in enumerate(test_loader):
        inputs,pos, label = data
        pos = pos.to(device)
        inputs = inputs.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model(inputs,pos) 
            loss = Loss_function(outputs, label)
            pmse = calculate_pmse(outputs, label)
            pmae = calculate_pmae(outputs, label)
            pmse_sum += pmse
            pmae_sum += pmae
            test_total_loss += loss.item()
            b,c = outputs.shape
            for batch in range(b):
                output = outputs[batch]
                label_batch = label[batch]
                for channel in range(c):
                    pre = output[channel].item()
                    gt = label_batch[channel].item()
                    prelist[channel].append(pre)
                    gtlist[channel].append(gt)

    prxlsit = []
    for i in range(len(prelist)):
        pre = prelist[i]
        gt = gtlist[i]
        prs,pv = pearsonr(pre,gt)
        prxlsit.append(prs)
    pcc = sum(prxlsit)/len(prxlsit)

    test_loss = test_total_loss/len(test_loader)
    pmse_mean = pmse_sum/len(test_loader)
    pmae_mean = pmae_sum/len(test_loader)
    mae = np.mean(pmae_mean)

    print(f"test loss:",test_loss,"test pcc:",pcc,"test pmae:",mae)


 