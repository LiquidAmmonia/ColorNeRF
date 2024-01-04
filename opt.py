import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    # base args
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--scene_name', type=str,
                        default='lego',
                        help='scene_name')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    # nerf args
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
        
    # training args
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained model weight to load (do not load optimizers, etc)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################
    # my exp args
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--read_gray', default=False, action="store_true",
                        help='Convert RGB images to grayscale during training \
                        excepct for the reference view')
    parser.add_argument('--loss_type', type=str, default='color',
                        help='scheduler type',
                        choices=['color', 'gray', 'gray_vgg'])
    parser.add_argument('--val_sanity_epoch', type=int, default=1,
                        help='val_sanity_epoch')
    # parser.add_argument('--ffr_dir', type=str,
    #                     default='/private_dataset/nerf_llff_data_images/nerf_llff_data_images.ffr',
    #                     help='root directory of dataset')
    parser.add_argument('--ffr_dir', type=str,
                        default='nerf_llff_data_bins/nerf_llff_data_images.ffr',
                        help='root directory of dataset')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers for dataloader')

    # coarse supervision args: DEPRECATED
    parser.add_argument('--use_coarse_rgb', default=False, action="store_true",
                        help='Use coarse rgbs as loss supervisions')
    parser.add_argument('--use_ref', default=False, action="store_true",
                        help='Use reference rgb image')

    # patch-based batching args 128 for ct2, 256 for lcoder and zhang
    parser.add_argument('--use_patch', default=True, action="store_true",
                        help='Use semantically-aware batching')
    parser.add_argument('--not_use_patch', dest='use_patch', action='store_false')
    parser.set_defaults(use_patch=True)
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Size of the patches')    
    parser.add_argument('--patch_sample_method', type=str, default='central',
                        help='patch_sample_method type',
                        choices=['central', 'random', 'full'])
    parser.add_argument('--create_pose_method', type=str, default='spheric',
                        help='straight for movie(paris)',
                        choices=['spheric', 'straight', 'spike_1', 'nir'])
    
    # teacher model args
    parser.add_argument('--teacher_model', type=str, default='ct2',
                        help='scheduler type',
                        choices=['ct2', 'lcoder', 'zhang'])
    ## lcoder args, only works when teacher_model == 'lcoder'
    parser.add_argument('--lcoder_caption', type=str,
                        default='a red flower in green leaves',
                        help='caption of the red flowers')
    ## zhang args, only works when teacher_model == 'zhang'
    parser.add_argument('--zhang_mode', type=str, default='fix',
                        help='scheduler type',
                        choices=['random', 'fix'])
    ## ct2 args, only works when teacher_model == 'ct2'
    parser.add_argument('--use_fintune_ct2', default=False, action="store_true",
                        help='Use fine-tuned ct2 model, only for Stage 2')
    parser.add_argument('--fintune_ct2_name', default='checkpoint_epoch_14_0211.pth', type=str,
                        help='Use fine-tuned ct2 model, in path')
    
    # ColorNeRF base args
    parser.add_argument('--train_stage', type=int, default=1,
                        help='training stage, 1 for grayscale training, 2 for color training')
    parser.add_argument('--normalize_illu', default=True, action="store_true",
                        help='In Stage 1, normalize illumination')
    parser.add_argument('--local_run', default=False, action="store_true",
                        help='NOT read from llff but from images hiden in nerf_llff_data_bins')
    parser.add_argument('--dense', default=False, action="store_true",
                        help='Use a model with more parameters')
    
    ## scene code: DEPRECATED
    parser.add_argument('--use_scene_code', default=False, action="store_true",
                        help='Use a learnable vector to represent scene identity')
    
    # colorloss args
    ## l2 loss on normalized ab
    parser.add_argument('--weight_l2_loss', type=float, default=1e-2,
                        help='weight of l2 loss on normalized ab')
    
    ## color classification loss
    parser.add_argument('--use_color_class_loss', action="store_true",
                        help='Use a learnable vector to represent scene identity')
    parser.add_argument('--not_use_color_class_loss', dest='use_color_class_loss', action='store_false')
    parser.set_defaults(use_color_class_loss=True)
    parser.add_argument('--color_class_start_epoch', type=int, default=0,
                        help='number of epoch to start color classification loss')
    parser.add_argument('--weight_color_class', type=float, default=1e-1,
                        help='weight of color classification loss')
    parser.add_argument('--color_class_T', type=float, default=0,
                        help='weight of color classification')
    parser.add_argument('--class_rebal_lambda', type=float, default=0.5,
                        help='weight of color classification')
    
    ## color edge loss
    parser.add_argument('--use_edge_loss', action="store_true",
                        help='Use a learnable vector to represent scene identity')
    parser.add_argument('--not_use_edge_loss', dest='use_edge_loss', action='store_false')
    parser.set_defaults(use_edge_loss=False)
    parser.add_argument('--weight_edge_loss', type=float, default=1e-3,
                        help='weight of color classification loss')
    parser.add_argument('--edge_start_epoch', type=int, default=0,
                        help='number of epoch to start color classification loss')
    
    # color histogram args
    parser.add_argument('--use_color_hist', action='store_true',
                        help='Use a learnable vector to represent scene identity')
    parser.add_argument('--not_use_color_hist', dest='use_color_hist', action='store_false')
    parser.set_defaults(use_color_hist=True)
    parser.add_argument('--manual_color_hist_register', type=str, default='null',
                        help='Manually reigister color hist image from file',)
    parser.add_argument('--color_hist_thres', type=float, default=0.92,
                        help='color_hist_thres')
    parser.add_argument('--num_color_hist', type=int, default=5,
                        help='color_hist_thres')
    parser.add_argument('--color_hist_force_accept', action='store_true',
                        help='Use a learnable vector to represent scene identity')
    parser.add_argument('--not_color_hist_force_accept', dest='color_hist_force_accept', action='store_false')
    parser.set_defaults(color_hist_force_accept=True)
    parser.add_argument('--color_hist_verbose', type=bool, default=False,)
    ## TV Loss
    parser.add_argument('--use_tv_loss', action="store_true",
                        help='Use a learnable vector to represent scene identity')
    parser.add_argument('--not_use_tv_loss', dest='use_tv_loss', action='store_false')
    parser.set_defaults(use_tv_loss=True)
    parser.add_argument('--weight_tv_loss', type=float, default=0.1,
                        help='weight of tv loss')
    
    return parser.parse_args()