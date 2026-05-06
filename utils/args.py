import argparse

def parser_args():
    parser = argparse.ArgumentParser()

    # ========================= federated learning parameters ========================
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--save_dir', type=str, default='../saved_mia_models',
                        help='saving path')
    parser.add_argument('--log_folder_name', type=str, default='/training_log_correct_iid/',
                        help='saving path')
    parser.add_argument('--samples_per_user', type=int, default=5000,
                        help="the number of samples in per user")
    parser.add_argument('--defense', type=str, default="none",
                        help="defense method:[mix_up, instahide, quant, sparse]")
    parser.add_argument('--d_scale', type=float, default=0.0,
                        help="param of quant or sparse defense")
    parser.add_argument('--klam', type=int, default=3,
                        help="the param for instahide scheme")
    parser.add_argument('--up_bound', type=float, default=0.65,
                        help="the up bound of lambda in instahide")
    parser.add_argument('--mix_alpha', type=float, default=0.01,
                        help="the param of beta distribution in mix up")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed")
    parser.add_argument('--frac', type=float, default=1,
                        help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=100,
                        help="local batch size: B")
    parser.add_argument('--lr_outer', type=float, default=1,
                        help="learning rate")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate for inner update")
    parser.add_argument('--lr_up', type=str, default='common',
                        help='optimizer: [common, milestone, cosine]')
    parser.add_argument('--schedule_milestone', type=list, default=[],
                         help="schedule lr")
    parser.add_argument('--gamma', type=float, default=0.99,
                         help="exponential weight decay")
    parser.add_argument('--iid', type=int,  default =1,
                        help='dataset is split iid or not')
    parser.add_argument('--MIA_mode', type=int,  default =1,
                        help='MIA score is computed or not')
    parser.add_argument('--beta', type=float, default=1,
                        help='Non-iid Dirichlet param')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer: [sgd, adam]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='communication round')
    parser.add_argument('--sampling_type', choices=['poisson', 'uniform'],
                         default='uniform', type=str,
                         help='which kind of client sampling we use') 
    parser.add_argument('--data_augment', type=int,  default =0,
                        help='data_augment')
    parser.add_argument('--sampling_proportion', type=float,  default=1.0,
                        help='sampling_proportion')
    parser.add_argument('--lira_attack', action='store_true', default=True,
                        help='lira_attack')
    parser.add_argument('--cosine_attack', action='store_true', default=True,
                        help='cosine_attack')
    
    # ============================ Model arguments ===================================
    parser.add_argument('--model_name', type=str, default='alexnet', choices=['alexnet','ResNet18','binn'],
                        help='model architecture name')
    parser.add_argument('--binn_input_size', type=int, default=128,
                        help='number of input gene/features for BINN')
    parser.add_argument('--binn_hidden_layers', type=str, default='128,64,32',
                        help='comma-separated BINN hidden layer widths')
    parser.add_argument('--binn_output_size', type=int, default=1,
                        help='BINN output size; 1 means binary BCEWithLogits training')
    parser.add_argument('--binn_dropout', type=float, default=0.2,
                        help='dropout probability for BINN hidden activations')
    parser.add_argument('--binn_activation', type=str, default='relu',
                        help='BINN hidden activation')
    parser.add_argument('--binn_output_last_layers', type=int, default=1,
                        help='number of final BINN layers used by residual readout')
    parser.add_argument('--binn_synthetic_samples', type=int, default=2000,
                        help='number of generated BINN training samples')
    parser.add_argument('--binn_synthetic_test_samples', type=int, default=500,
                        help='number of generated BINN test samples')
    parser.add_argument('--binn_synthetic_signal', type=float, default=2.0,
                        help='signal strength for generated BINN labels')
    
    parser.add_argument('--dataset', type=str, default='cifar100', help="name of dataset")
    
    parser.add_argument('--data_root', default='./Data',
                        help='dataset directory')

    # =========================== Other parameters ===================================
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--bp_interval', default=30, type=int, help='interval for starting bp the local part')
    parser.add_argument('--log_interval', default=1, type=int,
                        help='interval for evaluating loss and accuracy')
    parser.add_argument('--log_level', default='INFO', type=str,
                        help='logging level: DEBUG, INFO, WARNING, ERROR')
    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')
    parser.add_argument("--sigma_sgd",
        type=float,
        default=0.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
    "--grad_norm",
    type=float,
    default=1e4,
    help="Clip per-sample gradients to this norm",
    )
    # =========================== DP ===================================
    parser.add_argument('--dp', action='store_true', default=False,
                        help='whether dp')
    parser.add_argument('--sigma',  type=float, default= 0.1 , help='the sgd of Gaussian noise')
    args = parser.parse_args()

    return args
