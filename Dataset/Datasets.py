from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class AnnDataBatchDataset(torch.utils.data.Dataset):
    def __init__(self, adata, batch_size=32):
        self.adata = adata
        self.batch_size = batch_size
        self.num_cells = adata.shape[0]

        self.cell_expression = torch.tensor(adata.X.toarray()).to('cuda')
        self.condition = adata.obs['condition']

    def __len__(self):
        return self.num_cells

    def __getitem__(self, idx):
        cell_expression = self.cell_expression[idx, :]
        cell_condition = self.condition.iloc[idx]  # obtain cell's condition

        return cell_expression, cell_condition



class SourceModelDataset(Dataset):
    def __init__(self, adata, cell_list):
        self.cell_expression_all = torch.tensor(adata.X.toarray()).to('cuda')
        self.cell_list = cell_list
        self.cell_type_all = adata.obs['cell_type'].values.tolist()

        self.ctrl_base_list={}
        for cell_type in self.cell_list:
            current_type_adata=adata[adata.obs['cell_type'] == cell_type]
            current_type_adata=current_type_adata[current_type_adata.obs['condition']=='ctrl']
            mean_gene_expr=torch.mean(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            self.ctrl_base_list[cell_type]=mean_gene_expr

    def __len__(self):
        return self.cell_expression_all.shape[0]

    def __getitem__(self, idx):
        cell_expression = self.cell_expression_all[idx, :]
        if isinstance(idx, list):
            cell_type = [self.cell_type_all[i] for i in idx]
        else:
            cell_type = [self.cell_type_all[idx]]

        indices=[]
        ctrl_base=[]
        for ctype in cell_type:
            for i,label in enumerate(self.cell_list):
                if ctype==label:
                    indices.append(i)
            ctrl_base.append(self.ctrl_base_list[ctype])
        # indices = [ [i for i, label in enumerate(self.cell_list) if label == ctype] for ctype in cell_type ]
        ctrl_base=torch.stack(ctrl_base).to('cuda').squeeze()
        indices_tensor = torch.tensor(indices).to('cuda').squeeze()
        return {
            'feature':cell_expression,
            'cell_type':indices_tensor,
            # 'ctrl_base':ctrl_base,
        }

# obtain
class TargetModelDataset_Molecular(Dataset):
    def __init__(self,adata,adata_ctrl,cell_list,mole_embed,mole_list):
        self.adata=adata
        self.adata.obs.loc[self.adata.obs['condition'] == 'ctrl', 'dose_val'] = 0

        self.cell_expression_all = torch.tensor(adata.X.toarray()).to('cuda')
        self.cell_list = cell_list
        self.cell_type_all = adata.obs['cell_type'].values.tolist()
        self.SMILES_all = adata.obs['SMILES'].values.tolist()
        self.mole_embed=torch.tensor(mole_embed).to('cuda')
        self.dosage=torch.tensor(adata.obs['dose_val'].values.tolist())
        self.mole_list=mole_list # store a list of molecular

        self.ctrl_mean_list={}
        self.ctrl_var_list={}
        for cell_type in self.cell_list:
            current_type_adata=adata_ctrl[adata_ctrl.obs['cell_type'] == cell_type]
            current_type_adata=current_type_adata[current_type_adata.obs['condition']=='ctrl']
            mean_gene_expr=torch.mean(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            var_gene_expr= torch.var(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            self.ctrl_mean_list[cell_type]=mean_gene_expr
            self.ctrl_var_list[cell_type]=var_gene_expr


    def __len__(self):
        return self.adata.shape[0]

    # def __getitem__(self, idx):
    #     cell_expression = self.cell_expression_all[idx, :]
    #     if isinstance(idx, list):
    #         cell_type = [self.cell_type_all[i] for i in idx]
    #         molecular = [self.SMILES_all[i] for i in idx]
    #         dosage = self.dosage[idx]
    #     else:
    #         cell_type = [self.cell_type_all[idx]]
    #         molecular = [self.SMILES_all[idx]]
    #         dosage = self.dosage[idx].unsqueeze(0)
    #
    #     molecular = [s.split("|")[0] for s in molecular if s != '']
    #
    #     indices=[]
    #     ctrl_base=[]
    #     for ctype in cell_type:
    #         for i,label in enumerate(self.cell_list):
    #             if ctype==label:
    #                 indices.append(i)
    #         ctrl_base.append(self.ctrl_base_list[ctype])
    #
    #     indices_tensor = torch.tensor(indices).to('cuda')
    #
    #     indices_mole=[]
    #
    #     for mole in molecular:
    #         for i,smile in enumerate(self.mole_list):
    #             if smile==mole:
    #                 indices_mole.append(i)
    #
    #     mole_embed=self.mole_embed[indices_mole]
    #     ctrl_base=torch.stack(ctrl_base).to('cuda').squeeze()
    #     return {
    #         'cell_type':indices_tensor.to(torch.long) ,
    #         'feature':cell_expression,
    #         'mole':mole_embed.squeeze(),
    #         'dosage':dosage,
    #         'ctrl_base':ctrl_base,
    #     }

    def __getitem__(self, idx):
        cell_expression = self.cell_expression_all[idx, :]

        if isinstance(idx, list):
            cell_type = [self.cell_type_all[i] for i in idx]
            molecular = [self.SMILES_all[i] for i in idx]
            dosage = self.dosage[idx]
        else:
            cell_type = [self.cell_type_all[idx]]
            molecular = [self.SMILES_all[idx]]
            dosage = self.dosage[idx].unsqueeze(0)

        # 处理 molecular：控制组是空字符串或 None
        molecular_cleaned = []
        indices_mole = []
        for s in molecular:
            if s == 'CS(C)=O':
                molecular_cleaned.append(None)
                indices_mole.append(-1)
            else:
                mol = s.split("|")[0]
                molecular_cleaned.append(mol)
                for i, smile in enumerate(self.mole_list):
                    if mol == smile:
                        indices_mole.append(i)

        indices = []
        ctrl_mean = []
        ctrl_var = []
        for ctype in cell_type:
            for i, label in enumerate(self.cell_list):
                if ctype == label:
                    indices.append(i)
            ctrl_mean.append(self.ctrl_mean_list[ctype])
            ctrl_var.append(self.ctrl_var_list[ctype])

        indices_tensor = torch.tensor(indices).to('cuda')
        ctrl_mean = torch.stack(ctrl_mean).to('cuda').squeeze()
        ctrl_var = torch.stack(ctrl_var).to('cuda').squeeze()

        mole_embed = []
        for i in indices_mole:
            if i == -1:
                mole_embed.append(torch.zeros_like(self.mole_embed[0]))  # 全零嵌入
            else:
                mole_embed.append(self.mole_embed[i])
        mole_embed = torch.stack(mole_embed).to('cuda')


        return {
            'cell_type': indices_tensor.to(torch.long),
            'feature': cell_expression,
            'mole': mole_embed.squeeze(),
            'dosage': dosage,
            'ctrl_mean': ctrl_mean,
            'ctrl_var': ctrl_var,
            'x0':cell_expression,
        }


class TargetModelDataset_Gene(torch.utils.data.Dataset):
    def __init__(self, adata,adata_ctrl,cell_list,gene_list):
        self.gene_list=gene_list
        self.adata=adata
        self.cell_expression_all = torch.tensor(adata.X.toarray()).to('cuda')
        self.cell_list = cell_list
        self.cell_type_all = adata.obs['cell_type'].values.tolist()
        self.knockout_all = adata.obs['condition'].values.tolist()

        self.ctrl_mean_list={}
        self.ctrl_var_list={}
        for cell_type in self.cell_list:
            current_type_adata=adata_ctrl[adata_ctrl.obs['cell_type'] == cell_type]
            current_type_adata=current_type_adata[current_type_adata.obs['condition']=='ctrl']
            mean_gene_expr=torch.mean(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            var_gene_expr= torch.var(torch.tensor(current_type_adata.X.toarray()),dim=0).to('cuda')
            self.ctrl_mean_list[cell_type]=mean_gene_expr
            self.ctrl_var_list[cell_type]=var_gene_expr



    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        cell_expression = self.cell_expression_all[idx, :]
        if isinstance(idx, list):
            cell_type = [self.cell_type_all[i] for i in idx]
            knockout = [self.knockout_all[i] for i in idx]

        else:
            cell_type = [self.cell_type_all[idx]]
            knockout = [self.knockout_all[idx]]

        indices = []
        ctrl_mean=[]
        ctrl_var=[]
        for ctype in cell_type:
            for i, label in enumerate(self.cell_list):
                if ctype == label:
                    indices.append(i)
            ctrl_mean.append(self.ctrl_mean_list[ctype])
            ctrl_var.append(self.ctrl_var_list[ctype])

        indices_tensor = torch.tensor(indices).to('cuda')

        knockout_indices=[]
        for k in knockout:
            if k.split("+")[0]!='ctrl':
                knock_gene=k.split("+")[0]
            else:
                if len(k.split("+"))!=1:
                    knock_gene = k.split("+")[1]
                else:
                    knock_gene = 'ctrl'

            if knock_gene=='ctrl':
                knockout_indices.append(-1)
            else:
                knockout_indices.append(self.gene_list.index(knock_gene))

        knockout_indices = torch.tensor(knockout_indices).to('cuda')
        ctrl_mean=torch.stack(ctrl_mean).to('cuda').squeeze()
        ctrl_var=torch.stack(ctrl_var).to('cuda').squeeze()

        return {
            'cell_type': indices_tensor,
            'feature': cell_expression,
            'knockout': knockout_indices,
            'ctrl_mean': ctrl_mean,
            'ctrl_var': ctrl_var,
            'x0': cell_expression,
        }

def return_dataloader(adata,cell_type,adata_ctrl=None,mole_embed=None,mole_list=None,gene_name=None,source_model=True,pert_type="molecular",batch_size=32):
    if source_model:
        return DataLoader(SourceModelDataset(adata=adata,cell_list=cell_type),batch_size=batch_size,shuffle=True)
    else:
        if pert_type=="molecular":
            return DataLoader(TargetModelDataset_Molecular(adata,adata_ctrl=adata_ctrl,cell_list=cell_type,mole_embed=mole_embed,mole_list=mole_list),batch_size=batch_size,shuffle=True)
        else:
            return DataLoader(TargetModelDataset_Gene(adata,adata_ctrl=adata_ctrl,cell_list=cell_type,gene_list=gene_name),batch_size=batch_size,shuffle=True)