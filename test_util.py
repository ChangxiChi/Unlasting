import torch
from train import *
import json
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from model import *

def compute_rmse(y_true, y_pred):
    """
    Compute RMSE between the mean of y_true and y_pred.
    y_true: shape (n_samples_1, n_genes)
    y_pred: shape (n_samples_2, n_genes)
    """
    mean_true = np.mean(y_true, axis=0)  # shape: (n_genes,)
    mean_pred = np.mean(y_pred, axis=0)  # shape: (n_genes,)
    rmse = np.sqrt(np.mean((mean_true - mean_pred) ** 2))
    return rmse

def predict(pert_data,model_path,source_path,args,from_source=False,latent_norm=False,ablation="",pred_ood=False,sample_num=100):
    model, diffusion = create_model_and_diffusion(
        GRN=pert_data.GRN,
        cell_type_num=pert_data.cell_type_num,
        pert_type=args.pert_type,
        data_name=args.data_name,
        gene_num=args.gene_num,
        time_pos_dim=args.time_pos_dim,
        gene_wise_embed_dim=args.gene_wise_embed_dim,
        time_embed_dim=args.time_embed_dim,
        cell_type_embed_dim=args.cell_type_embed_dim,
        load_trained_source_model=args.load_trained_source_model,
        source_trainable=args.source_trainable,
        gene_init_dim=args.gene_init_dim,
        mole_dim=args.mole_dim,
        use_x_l=args.use_x_l,
        use_gwf=args.use_gwf,
        use_ggf=args.use_ggf,
        learn_sigma=args.learn_sigma,
        diffusion_steps=args.diffusion_steps,
        timestep_respacing=args.timestep_respacing,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        source_model=args.source_model,
    )

    model.to(dist_util.dev())
    model.load_state_dict(torch.load(model_path))

    model.eval()
    # source_model.eval()
    prediction={}

    mask_model=MaskModel(
        gene_num=args.gene_num,
        GRN=pert_data.GRN,
        cell_type_num=pert_data.cell_type_num,
        data_name=args.data_name,
        mole_dim=args.mole_dim,
        pert_type=args.pert_type,
    ).to('cuda')

    mask_model.eval()

    ctrl_base_list = {}
    for cell_type in pert_data.cell_type:
        current_type_adata = pert_data.train_cell_control[pert_data.train_cell_control.obs['cell_type'] == cell_type]
        current_type_adata = current_type_adata[current_type_adata.obs['condition'] == 'ctrl']
        mean_gene_expr = torch.mean(torch.tensor(current_type_adata.X.toarray()), dim=0).to('cuda')
        ctrl_base_list[cell_type] = mean_gene_expr

    if from_source==False:
        if args.pert_type=='molecular':
            for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
                current_test_cell=pert_data.adata[pert_data.adata.obs['condition'] == cond]
                current_cell_type=list(current_test_cell.obs['cell_type'].unique())

                mole=current_test_cell.obs['SMILES'].unique()[0]
                mole=mole.split("|")[0]
                mole_idx=pert_data.mole.index(mole)
                mole_embed=pert_data.mole_embed[mole_idx]

                for cell_type in current_cell_type:
                    specific_type_cell=current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                    all_dosage=list(specific_type_cell.obs['dose_val'].unique())
                    cell_type_idx=pert_data.cell_type.index(cell_type)
                    for dosage in all_dosage:
                        specific_type_dosage_cell=specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]
                        if sample_num==None:
                            sample_num=specific_type_dosage_cell.shape[0]
                        micro_cond = {
                            'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num,1),
                            'dosage': torch.tensor(dosage,dtype=torch.float32).to('cuda').repeat(sample_num,1),
                            'mole': mole_embed.repeat(sample_num,1),
                        }
                        predict_gene_expression=diffusion.ddim_sample_loop(
                            model=model,
                            shape=(sample_num,specific_type_dosage_cell.shape[1]),
                            model_kwargs=micro_cond,
                        )



                        key=args.pert_type+"_"+cond+"_"+cell_type+"_"+str(dosage)
                        prediction[key]=predict_gene_expression.cpu().numpy().tolist()

        else:
            for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
                current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())
                for cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                    cell_type_idx = pert_data.cell_type.index(cell_type)

                    if sample_num==None:
                        sample_num=specific_type_cell.shape[0]

                    knockout_gene_idx=pert_data.gene_name.index(cond)
                    micro_cond = {
                        'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1),
                        'knockout':torch.tensor(knockout_gene_idx).to('cuda').repeat(sample_num, 1),
                    }
                    predict_gene_expression = diffusion.ddim_sample_loop(
                        model=model,
                        shape=(sample_num,specific_type_cell.shape[1]),
                        model_kwargs=micro_cond,
                    )
                    key = args.pert_type + "_" + cond + "_" + cell_type
                    prediction[key] = predict_gene_expression.cpu().numpy().tolist()

        store_path = "./result/prediction/"+ ablation + args.data_name + ".json"
        with open(store_path, 'w') as f:
            json.dump(prediction, f)

    else:
        if args.pert_type == 'molecular':

            ctrl_dict={}
            latent_dict={}
            cell_types=pert_data.train_cell_control.obs['cell_type'].unique().tolist()
            for cell_type in cell_types:
                cells_ctrl=pert_data.train_cell_control[pert_data.train_cell_control.obs['cell_type'] == cell_type]
                random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')
                cell_type_idx = pert_data.cell_type.index(cell_type)
                kwargs = {
                    # 'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(1),
                    'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num),
                    'dosage': torch.tensor(0).to('cuda').repeat(sample_num),
                    # 'ctrl_base': ctrl_base_list[cell_type].to('cuda').repeat(sample_num),
                }
                latent = diffusion.ddim_reverse_sample_loop(
                    model=model,
                    image=gene_expr_ctrl,
                    # image=cell_ctrl_mean,
                    model_kwargs=kwargs,
                    clip_denoised=True,
                )

                latent_dict[cell_type] = latent
                ctrl_dict[cell_type] = gene_expr_ctrl


            if args.data_name == 'sciplex3':
                if pred_ood==False:
                    '''unseen drug covariate'''
                    test_cells=pert_data.adata[pert_data.adata.obs['unlasting_split']=='test']
                    test_smiles=list(test_cells.obs['SMILES'].unique())


                    for smiles in tqdm(test_smiles, desc="Unseen drug covariate conditions"):
                        current_test_cell = test_cells[test_cells.obs['SMILES'] == smiles]
                        current_cell_type = list(current_test_cell.obs['cell_type'].unique())

                        mole = smiles.split("|")[0]
                        mole_idx = pert_data.mole.index(mole)
                        mole_embed = pert_data.mole_embed[mole_idx:mole_idx+1]
                        mole_embed = torch.tensor(mole_embed).to('cuda')

                        for cell_type in current_cell_type:
                            specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                            all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                            cell_type_idx = pert_data.cell_type.index(cell_type)

                            cell_ctrl = pert_data.train_cell_control[
                                pert_data.train_cell_control.obs['cell_type'] == cell_type]

                            # cell_ctrl_mean=torch.mean(torch.tensor(cell_ctrl.X),dim=0).to('cuda').unsqueeze(0)
                            # random_indices = np.random.choice(cell_ctrl.n_obs, size=sample_num, replace=False)
                            # gene_expr_ctrl = torch.tensor(cell_ctrl[random_indices].X.toarray()).to('cuda')
                            #
                            # kwargs = {
                            #     # 'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(1),
                            #     'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num),
                            #     # 'ctrl_base': ctrl_base_list[cell_type].to('cuda').repeat(sample_num),
                            # }
                            # latent = diffusion.ddim_reverse_sample_loop(
                            #     model=model,
                            #     image=gene_expr_ctrl,
                            #     # image=cell_ctrl_mean,
                            #     model_kwargs=kwargs,
                            # )



                            for dosage in all_dosage:
                                specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]
                                if sample_num is None:
                                    sample_num = specific_type_dosage_cell.shape[0]

                                micro_cond = {
                                    'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1),
                                    'dosage': torch.tensor(dosage, dtype=torch.float32).to('cuda').repeat(sample_num, 1),
                                    'mole': mole_embed.repeat(sample_num, 1),
                                    'ctrl_mean': ctrl_dict[cell_type],
                                }

                                predict_gene_expression = diffusion.ddim_sample_loop(
                                    # noise=latent.expand(sample_num, -1),
                                    noise=latent_dict[cell_type],
                                    model=model,
                                    shape=(sample_num, specific_type_dosage_cell.shape[1]),
                                    # shape=specific_type_dosage_cell.shape,
                                    model_kwargs=micro_cond,
                                    eta=0,
                                    clip_denoised=True,
                                )



                                key = args.pert_type + "_" + smiles + "_" + cell_type + "_" + str(dosage)
                                prediction[key] = predict_gene_expression.cpu().numpy().tolist()

                else: # only predict OOD drugs
                    for cond in tqdm(pert_data.ood_cond, desc="OOD drugs conditions"):
                        current_test_cell = pert_data.adata[pert_data.adata.obs['SMILES'] == cond]
                        current_cell_type = list(current_test_cell.obs['cell_type'].unique())

                        mole = current_test_cell.obs['SMILES'].unique()[0]
                        mole = mole.split("|")[0]
                        mole_idx = pert_data.mole.index(mole)
                        mole_embed = pert_data.mole_embed[mole_idx]

                        for cell_type in current_cell_type:
                            specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                            all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                            cell_type_idx = pert_data.cell_type.index(cell_type)

                            for dosage in all_dosage:
                                specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]
                                # sample_num = specific_type_dosage_cell.shape[0]

                                micro_cond = {
                                    'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1),
                                    'dosage': torch.tensor(dosage, dtype=torch.float32).to('cuda').repeat(sample_num, 1),
                                    'mole': mole_embed.repeat(sample_num, 1),
                                    'ctrl_mean': ctrl_dict[cell_type]
                                }
                                predict_gene_expression = diffusion.ddim_sample_loop(
                                    noise=latent_dict[cell_type],
                                    model=model,
                                    shape=(sample_num, specific_type_dosage_cell.shape[1]),
                                    # shape=specific_type_dosage_cell.shape,
                                    model_kwargs=micro_cond,
                                    eta=0,
                                    clip_denoised=True,
                                )
                                key = args.pert_type + "_" + cond + "_" + cell_type + "_" + str(dosage)
                                prediction[key] = predict_gene_expression.cpu().numpy().tolist()

        else:
            latent_dict={}
            ctrl_dict={}
            cell_types=pert_data.train_cell_control.obs['cell_type'].unique().tolist()
            for cell_type in cell_types:
                cells_ctrl=pert_data.train_cell_control[pert_data.train_cell_control.obs['cell_type'] == cell_type]
                random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')
                cell_type_idx = pert_data.cell_type.index(cell_type)
                kwargs = {
                    # 'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(1),
                    'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num),
                    'knockout': torch.tensor(-1).to('cuda').repeat(sample_num)
                    # 'ctrl_base': ctrl_base_list[cell_type].to('cuda').repeat(sample_num),
                }
                latent = diffusion.ddim_reverse_sample_loop(
                    model=model,
                    image=gene_expr_ctrl,
                    # image=cell_ctrl_mean,
                    model_kwargs=kwargs,
                    clip_denoised=False,
                )
                latent_dict[cell_type] = latent
                ctrl_dict[cell_type] = gene_expr_ctrl
            if args.data_name=="adamson":
                for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
                    current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
                    current_cell_type = list(current_test_cell.obs['cell_type'].unique())
                    for cell_type in current_cell_type:
                        specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]

                        # sample_num=specific_type_cell.shape[0]
                        # cells_ctrl = pert_data.train_cell_control[
                        #     pert_data.train_cell_control.obs['cell_type'] == cell_type]
                        # random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                        # gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')
                        # cell_type_idx = pert_data.cell_type.index(cell_type)
                        # kwargs = {
                        #     'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num),
                        #     'knockout': torch.tensor(-1).to('cuda').repeat(sample_num)
                        # }
                        # latent = diffusion.ddim_reverse_sample_loop(
                        #     model=model,
                        #     image=gene_expr_ctrl,
                        #     # image=cell_ctrl_mean,
                        #     model_kwargs=kwargs,
                        #     clip_denoised=False,
                        # )

                        # latent_dict[cell_type] = latent
                        # ctrl_dict[cell_type] = gene_expr_ctrl


                        cell_type_idx = pert_data.cell_type.index(cell_type)
                        knockout_gene_idx=pert_data.gene_name.index(cond)
                        micro_cond = {
                            'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1),
                            'knockout': torch.tensor(knockout_gene_idx).to('cuda').repeat(sample_num, 1),
                            # 'ctrl_mean': gene_expr_ctrl
                            'ctrl_mean':ctrl_dict[cell_type]
                            # 'ctrl_base': ctrl_base_list[cell_type].to('cuda').repeat(sample_num, 1),
                        }

                        predict_gene_expression = diffusion.ddim_sample_loop(
                            model=model,
                            noise=latent_dict[cell_type],
                            # noise=latent,
                            shape=(sample_num, specific_type_cell.shape[1]),
                            model_kwargs=micro_cond,
                            eta=0,
                            clip_denoised=False,
                        )

                        # mask_prob = mask_model.predict(x=ctrl_dict[cell_type],
                        #                                knockout=torch.tensor(knockout_gene_idx).to('cuda').repeat(
                        #                                    sample_num, 1),
                        #                                cell_type=torch.tensor(cell_type_idx).to('cuda').repeat(
                        #                                    sample_num, 1),)
                        #
                        # mask=(mask_prob > 0.5)
                        # predict_gene_expression=predict_gene_expression*mask

                        key = args.pert_type + "_" + cond + "_" + cell_type
                        prediction[key] = predict_gene_expression.cpu().numpy().tolist()

            elif args.data_name=="norman":
                for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
                    current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
                    current_cell_type = list(current_test_cell.obs['cell_type'].unique())
                    for cell_type in current_cell_type:
                        specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                        cell_type_idx = pert_data.cell_type.index(cell_type)

                        knockout_gene_idx = [pert_data.gene_name.index(cond.split('+')[0]),pert_data.gene_name.index(cond.split('+')[1])]

                        micro_cond = {
                            'cell_type': torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1),
                            'knockout': torch.tensor(knockout_gene_idx).to('cuda').repeat(sample_num, 1),
                            'ctrl_mean': ctrl_dict[cell_type],
                            'single': False,
                            # 'ctrl_base': ctrl_base_list[cell_type].to('cuda').repeat(sample_num, 1),
                        }

                        predict_gene_expression = diffusion.ddim_sample_loop(
                            model=model,
                            noise=latent_dict[cell_type],
                            shape=(sample_num, specific_type_cell.shape[1]),
                            model_kwargs=micro_cond,
                            eta=0,
                            clip_denoised=False,
                        )

                        key = args.pert_type + "_" + cond + "_" + cell_type
                        prediction[key] = predict_gene_expression.cpu().numpy().tolist()

        if args.data_name=='sciplex3' and pred_ood==True:
            store_path = "./result/prediction/" + ablation + args.data_name + "_from_source_only_ood.json"
        else:
            store_path = "./result/prediction/"+ ablation + args.data_name + "_from_source.json"
        with open(store_path, 'w') as f:
            json.dump(prediction, f)

    return prediction

def get_perturb_info_emb_dict(pert_data,model_path,args):
    model, diffusion = create_model_and_diffusion(
        GRN=pert_data.GRN,
        cell_type_num=pert_data.cell_type_num,
        pert_type=args.pert_type,
        data_name=args.data_name,
        gene_num=args.gene_num,
        time_pos_dim=args.time_pos_dim,
        gene_wise_embed_dim=args.gene_wise_embed_dim,
        time_embed_dim=args.time_embed_dim,
        cell_type_embed_dim=args.cell_type_embed_dim,
        load_trained_source_model=args.load_trained_source_model,
        source_trainable=args.source_trainable,
        gene_init_dim=args.gene_init_dim,
        mole_dim=args.mole_dim,
        use_x_l=args.use_x_l,
        use_gwf=args.use_gwf,
        use_ggf=args.use_ggf,
        learn_sigma=args.learn_sigma,
        diffusion_steps=args.diffusion_steps,
        timestep_respacing=args.timestep_respacing,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        source_model=args.source_model,
    )

    model.to(dist_util.dev())
    model.load_state_dict(torch.load(model_path))

    model.eval()
    perturb_info_emb_train={}
    perturb_info_emb_test={}
    with torch.no_grad():
        perturb_info_emb_ood = {}
        if args.pert_type=='molecular':
            train_cells = pert_data.adata[pert_data.adata.obs['unlasting_split'] == 'train']
            train_perturb_cond = train_cells.obs['cov_drug_dose_name'].unique().tolist()

            test_cells = pert_data.adata[pert_data.adata.obs['unlasting_split'] == 'test']
            test_perturb_cond = test_cells.obs['cov_drug_dose_name'].unique().tolist()

            ood_cells = pert_data.adata[pert_data.adata.obs['unlasting_split'] == 'ood']
            ood_perturb_cond = ood_cells.obs['cov_drug_dose_name'].unique().tolist()

            for cond in train_perturb_cond:
                cell_type=cond.split('_')[0]
                drug=cond.split('_')[1]
                dose=torch.tensor(float(cond.split('_')[2])).to('cuda')
                if drug=='control':
                    continue
                else:
                    smiles=train_cells[train_cells.obs['cov_drug_dose_name']==cond].obs['SMILES'][0]
                    smiles_idx=pert_data.mole.index(smiles)
                    smiles_emb=pert_data.mole_embed[smiles_idx]
                    smiles_emb=torch.tensor(smiles_emb).to('cuda')
                    emb=model.perturb_info_emb(mole=smiles_emb.unsqueeze(0),dosage=dose.unsqueeze(0),cell_type=torch.tensor(pert_data.cell_type.index(cell_type)).to('cuda').unsqueeze(0))
                    perturb_info_emb_train[cond]=emb

            for cond in test_perturb_cond:
                cell_type=cond.split('_')[0]
                drug=cond.split('_')[1]
                dose=torch.tensor(float(cond.split('_')[2])).to('cuda')
                if drug=='control':
                    continue
                else:
                    smiles=test_cells[test_cells.obs['cov_drug_dose_name']==cond].obs['SMILES'][0]
                    smiles_idx=pert_data.mole.index(smiles)
                    smiles_emb=pert_data.mole_embed[smiles_idx]
                    smiles_emb=torch.tensor(smiles_emb).to('cuda')
                    emb=model.perturb_info_emb(mole=smiles_emb.unsqueeze(0),dosage=dose.unsqueeze(0),cell_type=torch.tensor(pert_data.cell_type.index(cell_type)).to('cuda').unsqueeze(0))
                    perturb_info_emb_test[cond]=emb

            for cond in ood_perturb_cond:
                cell_type=cond.split('_')[0]
                drug=cond.split('_')[1]
                dose=torch.tensor(float(cond.split('_')[2])).to('cuda')
                if drug=='control':
                    continue
                else:
                    smiles=ood_cells[ood_cells.obs['cov_drug_dose_name']==cond].obs['SMILES'][0]
                    smiles_idx=pert_data.mole.index(smiles)
                    smiles_emb=pert_data.mole_embed[smiles_idx]
                    smiles_emb = torch.tensor(smiles_emb).to('cuda')
                    emb=model.perturb_info_emb(mole=smiles_emb.unsqueeze(0),dosage=dose.unsqueeze(0),cell_type=torch.tensor(pert_data.cell_type.index(cell_type)).to('cuda').unsqueeze(0))
                    perturb_info_emb_ood[cond]=emb

            return perturb_info_emb_train, perturb_info_emb_test,perturb_info_emb_ood

        else:
            train_cells = pert_data.train_cell
            train_perturb_cond = train_cells.obs['knockout'].unique().tolist()
            train_perturb_cond.remove('ctrl')

            test_cells = pert_data.adata[pert_data.adata.obs['knockout'].isin(pert_data.test_cond)]
            test_perturb_cond = pert_data.test_cond


            for cond in train_perturb_cond:
                if cond == 'ctrl':
                    continue
                else:
                    gene_idx = pert_data.gene_name.index(cond)
                    gene_idx=torch.tensor(gene_idx).to('cuda').unsqueeze(0)
                    emb = model.perturb_info_emb(knockout=gene_idx)
                    perturb_info_emb_train[cond] = emb

            for cond in test_perturb_cond:
                if cond == 'control':
                    continue
                else:
                    gene_idx = pert_data.gene_name.index(cond)
                    gene_idx=torch.tensor(gene_idx).to('cuda').unsqueeze(0)
                    emb = model.perturb_info_emb(knockout=gene_idx)
                    perturb_info_emb_test[cond] = emb

            return perturb_info_emb_train,perturb_info_emb_test,None


def gaussian_kernel(x, y, sigma=1.0):
    x = x[:, np.newaxis, :]  # (n,1,d)
    y = y[np.newaxis, :, :]  # (1,m,d)
    dist = np.sum((x - y) ** 2, axis=2)
    return np.exp(-dist / (2 * sigma ** 2))


def compute_mmd(P, Q, sigma=1.0):
    """
    Compute MMD between two arrays (n_samples, n_features)
    """
    K_PP = gaussian_kernel(P, P, sigma)
    K_QQ = gaussian_kernel(Q, Q, sigma)
    K_PQ = gaussian_kernel(P, Q, sigma)

    return K_PP.mean() + K_QQ.mean() - 2 * K_PQ.mean()

def compute_e_distance(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)

    n, m = X.shape[0], Y.shape[0]

    cross_term = np.mean(cdist(X, Y, metric='euclidean'))
    X_term = np.sum(pdist(X, metric='euclidean')) * 2 / (n * (n - 1))
    Y_term = np.sum(pdist(Y, metric='euclidean')) * 2 / (m * (m - 1))

    e_dist = 2 * cross_term - X_term - Y_term
    return np.sqrt(max(e_dist, 0))


def get_top_logfc_idx(treatment, control, topk,pseudocount=1e-6):
    treatment_mean = np.mean(treatment, axis=0)
    control_mean = np.mean(control, axis=0)
    logfc = np.log2((treatment_mean + pseudocount) / (control_mean + pseudocount))
    top_idx = np.argsort(-logfc)[:topk]
    return top_idx

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

import torch.nn.functional as F
from sklearn.decomposition import PCA

def filter(adata,train_perturb_emb_dict,query_emb,samples,k=5,sample_k=3,data_name='sciplex3'):
    samples = np.array(samples)
    train_metric=torch.stack(list(train_perturb_emb_dict.values()), dim=0)
    train_keys=list(train_perturb_emb_dict.keys())
    pca = PCA(n_components=64)
    pca.fit(train_metric.cpu())

    query_emb = pca.transform(query_emb.unsqueeze(0).cpu())
    train_metric = pca.transform(train_metric.cpu())
    query_emb=torch.tensor(query_emb)
    train_metric=torch.tensor(train_metric)
    cos_sim = F.cosine_similarity(train_metric, query_emb, dim=1)
    topk_sim, topk_idx = torch.topk(cos_sim, k=k, largest=True)
    top_sim_keys=[train_keys[idx] for idx in topk_idx]

    # pcc_delta_sum=np.zeros(samples.shape[0])
    # for cond in top_sim_keys:
    #     if data_name=='sciplex3':
    #         cell_type=cond.split('_')[0]
    #         ctrl_cells=adata[adata.obs['condition']=='ctrl']
    #         ctrl_mean=np.mean(ctrl_cells[ ctrl_cells.obs['cell_type'] == cell_type].X.toarray(),axis=0)
    #         true_mean=np.mean(adata[adata.obs['cov_drug_dose_name']==cond].X.toarray(),axis=0)
    #     elif data_name=='adamson':
    #         ctrl_cells = adata[adata.obs['condition'] == 'ctrl']
    #         ctrl_mean = np.mean(ctrl_cells.X.toarray(), axis=0)
    #         true_mean = np.mean(adata[adata.obs['knockout'] == cond].X.toarray(), axis=0)
    #
    #
    #     for i in range(samples.shape[0]):
    #         corr, _ = pearsonr(true_mean-ctrl_mean, samples[i]-ctrl_mean)
    #         pcc_delta_sum[i]+=corr
    #
    # filter_idx = np.argsort(pcc_delta_sum)[-sample_k:][::-1]
    # filter_samples=[samples[i] for i in filter_idx]

    # mse_sum = np.zeros(samples.shape[0])
    dir_sim_sum = np.zeros(samples.shape[0])
    # norm_sum = np.zeros(samples.shape[0])
    # pcc_delta_sum = np.zeros(samples.shape[0])
    # cos_sum = np.zeros(samples.shape[0])

    for cond in top_sim_keys:
        if data_name == 'sciplex3':
            cell_type = cond.split('_')[0]
            ctrl_cells = adata[adata.obs['condition'] == 'ctrl']
            ctrl_mean = np.mean(ctrl_cells[ctrl_cells.obs['cell_type'] == cell_type].X.toarray(), axis=0)
            true_mean = np.mean(adata[adata.obs['cov_drug_dose_name'] == cond].X.toarray(), axis=0)
        elif data_name == 'adamson':
            ctrl_cells = adata[adata.obs['condition'] == 'ctrl']
            ctrl_mean = np.mean(ctrl_cells.X.toarray(), axis=0)
            true_mean = np.mean(adata[adata.obs['knockout'] == cond].X.toarray(), axis=0)

        target_delta = true_mean - ctrl_mean
        target_sign = np.sign(target_delta)
        for i in range(samples.shape[0]):
            pred_delta = samples[i] - ctrl_mean
            pred_sign = np.sign(pred_delta)

            match_ratio = np.mean(target_sign == pred_sign)
            dir_sim_sum[i] += match_ratio
            # norm_sum[i] += np.abs(np.linalg.norm(samples[i] - ctrl_mean) - np.linalg.norm(true_mean - ctrl_mean))
            # mse_sum[i] += mean_squared_error(samples[i] - ctrl_mean,true_mean - ctrl_mean)
            # corr, _ = pearsonr(true_mean - ctrl_mean, samples[i] - ctrl_mean)
            # pcc_delta_sum[i]+=corr
            # cos_sum[i] +=cosine_similarity(samples[i], target_delta)

    dir_top_idx = np.argsort(dir_sim_sum)[:sample_k]
    # mse_top_idx = np.argsort(mse_sum)[:sample_k]
    # norm_top_idx = np.argsort(norm_sum)[:sample_k]
    # pcc_delta_top_idx = np.argsort(pcc_delta_sum)[:sample_k]

    # top_idx_intersection = np.intersect1d(np.intersect1d(mse_top_idx, norm_top_idx), pcc_delta_top_idx)
    # top_idx_intersection = np.intersect1d(norm_top_idx, pcc_delta_top_idx)


    filter_samples = samples[dir_top_idx]

    return np.array(filter_samples)


import matplotlib.pyplot as plt

def plot_fit(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)

    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)

    x_min, x_max = y_true.min(), y_true.max()
    x_center = (x_max + x_min) / 2
    x_half_range = (x_max - x_min) / 2
    plt.xlim(x_center - 3 * x_half_range, x_center + 3 * x_half_range)

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs True')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


import umap
import matplotlib.pyplot as plt

def umap_simple(X, labels=None):
    reducer = umap.UMAP(random_state=42)
    X_emb = reducer.fit_transform(X)
    plt.figure(figsize=(6, 5))
    if labels is not None:
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=labels, cmap='tab10', s=10)
    else:
        plt.scatter(X_emb[:, 0], X_emb[:, 1], s=10)
    plt.title("UMAP")
    plt.tight_layout()
    plt.show()


def umap_two_arrays(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    combined = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))  # 0 for X, 1 for Y

    reducer = umap.UMAP(random_state=42)
    emb = reducer.fit_transform(combined)

    plt.figure(figsize=(6, 5))
    plt.scatter(emb[:len(X), 0], emb[:len(X), 1], label='X', alpha=0.6, s=10)
    plt.scatter(emb[len(X):, 0], emb[len(X):, 1], label='Y', alpha=0.6, s=10)
    plt.legend()
    plt.title("UMAP of X and Y")
    plt.tight_layout()
    plt.show()


from sklearn.cluster import KMeans
import numpy as np

def select_diverse_by_kmeans(samples, n_samples=100):
    kmeans = KMeans(n_clusters=n_samples, random_state=0).fit(samples)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    selected = []
    for i in range(n_samples):
        cluster_points = np.where(labels == i)[0]
        if len(cluster_points) == 0:
            continue
        center = centers[i]
        dists = np.linalg.norm(samples[cluster_points] - center, axis=1)
        selected.append(cluster_points[np.argmin(dists)])
    return samples[selected]

def select_diverse_by_avg_dist(X, n_samples=100):
    D = cdist(X, X, metric='euclidean')
    avg_dists = D.mean(axis=1)
    selected_idx = np.argsort(avg_dists)[-n_samples:]
    return X[selected_idx]

from sklearn.metrics import pairwise_distances

def farthest_point_sampling(X, n_samples=100):
    n = X.shape[0]
    selected_indices = [np.random.randint(0, n)]
    distances = pairwise_distances(X, X[selected_indices])

    for _ in range(1, n_samples):
        min_distances = distances.min(axis=1)
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
        dist_new = pairwise_distances(X, X[[next_idx]])
        distances = np.minimum(distances, dist_new)

    return X[selected_indices]


def generate_binary_samples_numpy(prob_list, n_samples):
    prob_array = np.array(prob_list).reshape(1, -1)
    rand_vals = np.random.rand(n_samples, len(prob_list))
    samples = (rand_vals < prob_array).astype(int)
    return samples