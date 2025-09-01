from model import MaskModel
from train import *
from Dataset.Preprocess import *
from Dataset.Datasets import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


args_train = parse_args()

# target model
# args_train.data_name = "sciplex3"
# args_train.pert_type = "molecular"
# args_train.gene_num=2000
pred_ood=False

args_train.data_name = "norman"
args_train.pert_type = "gene"
args_train.gene_num=5000

args_train.epoch=4
args_train.threshold = 0.3
args_train.threshold_co=0.3
args_train.batch_size=64

sample_num=500

args_train.resume_checkpoint = "./result/target_" + args_train.data_name + "_" + str(args_train.gene_num)

if not os.path.exists(args_train.resume_checkpoint):
    os.makedirs(args_train.resume_checkpoint)

print('**************training args*************')

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

if args_train.pert_type=='molecular':
    pert_data=PertData(hvg_num=args_train.gene_num,pert_type=args_train.pert_type,data_name=args_train.data_name,
                       path=current_dir,threshold=args_train.threshold,threshold_co=args_train.threshold_co)
else:
    pert_data=PertData(hvg_num=args_train.gene_num,pert_type=args_train.pert_type,data_name=args_train.data_name,
                       path=current_dir,threshold=args_train.threshold,threshold_co=args_train.threshold_co)

mask_model=MaskModel(
    gene_num=args_train.gene_num,
    GRN=pert_data.GRN,
    cell_type_num=pert_data.cell_type_num,
    data_name=args_train.data_name,
    mole_dim=args_train.mole_dim,
    pert_type=args_train.pert_type,
).to('cuda')

mask_model.load_state_dict(torch.load(args_train.resume_checkpoint+"/mask_model.pt"))
mask_model.eval()

mask_prediction={}

with torch.no_grad():
    if args_train.pert_type=="gene":
        if args_train.data_name=="adamson":
            for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
                current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())
                for cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                    cell_type_idx = pert_data.cell_type.index(cell_type)

                    cell_ctrl = pert_data.train_cell_control[pert_data.train_cell_control.obs['cell_type'] == cell_type]
                    random_indices = np.random.choice(cell_ctrl.n_obs, size=sample_num, replace=False)
                    gene_expr_ctrl = torch.tensor(cell_ctrl[random_indices].X.toarray()).to('cuda')

                    knockout_gene_idx=pert_data.gene_name.index(cond)
                    pred = mask_model.predict(x=gene_expr_ctrl,
                                              knockout=torch.tensor(knockout_gene_idx).to('cuda').repeat(sample_num, 1),
                                              cell_type=torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1))

                    pred=np.array(pred.cpu())

                    # pred_mask=np.any(pred,axis=0)

                    true = specific_type_cell.X > 0
                    true_expressed_genes_frequency = np.mean(true, axis=0)
                    # pred_expressed_genes_frequency = np.mean(pred, axis=0)

                    pred_mean=np.mean(pred,axis=0)
                    # pred[pred<0.01]=0


                    # mask = pred > pred_mean
                    # low_confidence_cols = pred_mean < 0.25
                    # mask[:, low_confidence_cols] = 0.

                    n, d = pred.shape
                    mask = np.zeros_like(pred)

                    for j in range(d):
                        col = pred[:, j]
                        p = pred_mean[j]

                        k = int(np.floor(p * n))
                        if k == 0:
                            continue
                        idx = np.argsort(col)[-k:]
                        mask[idx, j] = 1.

                    corr,_=pearsonr(true_expressed_genes_frequency, np.mean(mask,axis=0))
                    print(corr)

                    key = args_train.pert_type + "_" + cond + "_" + cell_type
                    # mask_prediction[key] = mask.tolist()
                    mask_prediction[key] = pred_mean.tolist()

        elif args_train.data_name=="norman":
            for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
                current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())
                for cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                    cell_type_idx = pert_data.cell_type.index(cell_type)

                    cell_ctrl = pert_data.train_cell_control[
                        pert_data.train_cell_control.obs['cell_type'] == cell_type]
                    random_indices = np.random.choice(cell_ctrl.n_obs, size=sample_num, replace=False)
                    gene_expr_ctrl = torch.tensor(cell_ctrl[random_indices].X.toarray()).to('cuda')

                    knockout_gene_idx = [pert_data.gene_name.index(cond.split('+')[0]),pert_data.gene_name.index(cond.split('+')[1])]

                    pred = mask_model.predict_double(x=gene_expr_ctrl,
                                              knockout=torch.tensor(knockout_gene_idx).to('cuda').repeat(sample_num,1),
                                              cell_type=torch.tensor(cell_type_idx).to('cuda').repeat(sample_num,1))

                    pred = np.array(pred.cpu())

                    # pred_mask=np.any(pred,axis=0)

                    true = specific_type_cell.X > 0
                    true_expressed_genes_frequency = np.mean(true, axis=0)
                    true_mask = (true_expressed_genes_frequency > 0.1)

                    pred_expressed_genes_frequency = np.mean(pred, axis=0)
                    pred_mask = (pred_expressed_genes_frequency > 0.1)

                    cosine_similarity(true_expressed_genes_frequency.reshape(1, -1),
                                      pred_expressed_genes_frequency.reshape(1, -1))
                    corr, _ = pearsonr(true_expressed_genes_frequency, pred_expressed_genes_frequency)
                    print(corr)

                    key = args_train.pert_type + "_" + cond + "_" + cell_type
                    mask_prediction[key] = pred_expressed_genes_frequency.tolist()



    elif args_train.pert_type=="molecular":
        if pred_ood:
            test_cells = pert_data.adata[pert_data.adata.obs['unlasting_split'] == 'ood']

        else:
            test_cells = pert_data.adata[pert_data.adata.obs['unlasting_split'] == 'test']

        test_smiles = list(test_cells.obs['SMILES'].unique())

        for smiles in tqdm(test_smiles, desc="Unseen drug covariate conditions"):
            current_test_cell = test_cells[test_cells.obs['SMILES'] == smiles]
            current_cell_type = list(current_test_cell.obs['cell_type'].unique())

            mole = smiles.split("|")[0]
            mole_idx = pert_data.mole.index(mole)
            mole_embed = pert_data.mole_embed[mole_idx:mole_idx + 1]
            mole_embed = torch.tensor(mole_embed).to('cuda')

            for cell_type in current_cell_type:
                specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                cell_type_idx = pert_data.cell_type.index(cell_type)

                cell_ctrl = pert_data.train_cell_control[
                    pert_data.train_cell_control.obs['cell_type'] == cell_type]

                random_indices = np.random.choice(cell_ctrl.n_obs, size=sample_num, replace=False)
                gene_expr_ctrl = torch.tensor(cell_ctrl[random_indices].X.toarray()).to('cuda')


                for dosage in all_dosage:
                    specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]
                    if sample_num is None:
                        sample_num = specific_type_dosage_cell.shape[0]

                    pred=mask_model.predict(x=gene_expr_ctrl,
                                            mole=mole_embed.repeat(sample_num, 1),
                                            dosage=torch.tensor(dosage, dtype=torch.float32).to('cuda').repeat(sample_num, 1),
                                            cell_type=torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1))

                    pred = np.array(pred.cpu())

                    # pred_mask=np.any(pred,axis=0)

                    true = specific_type_dosage_cell.X > 0
                    true_expressed_genes_frequency = np.mean(true, axis=0)
                    true_mask = (true_expressed_genes_frequency > 0.1)

                    pred_expressed_genes_frequency = np.mean(pred, axis=0)
                    pred_mask = (pred_expressed_genes_frequency > 0.1)

                    cosine_similarity(true_expressed_genes_frequency.reshape(1, -1),
                                      pred_expressed_genes_frequency.reshape(1, -1))
                    corr, _ = pearsonr(true_expressed_genes_frequency, pred_expressed_genes_frequency)
                    print(corr)

                    key = args_train.pert_type + "_" + smiles + "_" + cell_type + "_" + str(dosage)
                    mask_prediction[key] = pred_expressed_genes_frequency.tolist()

if args_train.data_name == 'sciplex3' and pred_ood == True:
    store_path = "./result/prediction/" + args_train.data_name + "_mask_only_ood.json"
else:
    store_path = "./result/prediction/" + args_train.data_name + "_mask.json"
with open(store_path, 'w') as f:
    json.dump(mask_prediction, f)