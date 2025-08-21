from train import *
import argparse
from Dataset.Preprocess import *
from Dataset.Datasets import *


if __name__ == "__main__":
    args_train = parse_args()

    # target model

    args_train.pert_type = "gene"
    args_train.data_name = "adamson"
    args_train.threshold = 0.3
    args_train.threshold_co= 0.3
    args_train.gene_num=5000
    args_train.log_interval=1e4
    args_train.save_interval=1e4
    args_train.lr_anneal_steps=1e4

    # args_train.pert_type = "sciplex3"
    # args_train.data_name = "molecular"
    # args_train.threshold = 0.3
    # args_train.threshold_co= 0.3
    # args_train.gene_num=2000
    # args_train.log_interval=1e4
    # args_train.save_interval=1e4
    # args_train.lr_anneal_steps=8e4

    args_train.load_trained_source_model=False
    args_train.source_model=False
    args_train.source_model_trainable = True
    args_train.predict_xstart = True
    args_train.batch_size=32

    args_train.logger_path = "./result/logs/target_" + args_train.data_name + "_" + str(args_train.gene_num)
    args_train.resume_checkpoint = "./result/target_" + args_train.data_name + "_" + str(args_train.gene_num)

    if not os.path.exists(args_train.resume_checkpoint):
        os.makedirs(args_train.resume_checkpoint)

    print('**************training args*************')
    print(args_train)
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    if args_train.pert_type=='molecular':
        pert_data=PertData(hvg_num=args_train.gene_num,pert_type=args_train.pert_type,data_name=args_train.data_name,path=current_dir)
        data=return_dataloader(pert_data.train_cell,adata_ctrl=pert_data.train_cell,cell_type=pert_data.cell_type,mole_embed=pert_data.mole_embed,mole_list=pert_data.mole,source_model=args_train.source_model,pert_type=args_train.pert_type,batch_size=args_train.batch_size)
    else:
        pert_data=PertData(hvg_num=args_train.gene_num,pert_type=args_train.pert_type,data_name=args_train.data_name,
                           path=current_dir,threshold=args_train.threshold,threshold_co=args_train.threshold_co)
        data=return_dataloader(pert_data.train_cell,adata_ctrl=pert_data.train_cell,cell_type=pert_data.cell_type,
                               gene_name=pert_data.gene_name,source_model=args_train.source_model,pert_type=args_train.pert_type,
                               batch_size=args_train.batch_size)
        # data=return_dataloader(pert_data.train_cell_treated,adata_ctrl=pert_data.train_cell,cell_type=pert_data.cell_type,gene_name=pert_data.gene_name,source_model=args_train.source_model,pert_type=args_train.pert_type,batch_size=args_train.batch_size)


    losses_target = run_training(data=data,cell_type_num=pert_data.cell_type_num,GRN=pert_data.GRN,args=args_train)