from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'ResNet50'
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# Value of colorjitter brightness
_C.INPUT.BRIGHTNESS = 0.0
# Value of colorjitter contrast
_C.INPUT.CONTRAST = 0.0
# Value of colorjitter saturation
_C.INPUT.SATURATION = 0.0
# Value of colorjitter hue
_C.INPUT.HUE = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('Market1501')
# Setup storage directroy for dataset
_C.DATASETS.STORE_DIR = ('/home/xiaocaibi/PSM_DATA')
#_C.DATASETS.STORE_DIR = ('/mnt/home/reid_stargan30_iter8000/DATA')
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 8

_C.DATALOADER.SHUFFLE = True

_C.DATALOADER.METHOD = 'default'

_C.DATALOADER.NUM_JUMP = 1
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.7

_C.SOLVER.MARGIN = 0.3

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.GAMMA = 0.1

_C.SOLVER.STEP = 40

_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.EVAL_PERIOD = 50
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 8

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.LOAD_EPOCH = 120


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.DEVICE = "cuda:0"
_C.OUTPUT_DIR = ""
_C.RE_RANKING = False

_C.img_size=[256,128]
_C.num_domains=6
_C.latent_dim=16
_C.hidden_dim=512
_C.style_dim=64
_C.lambda_reg=1
_C.lambda_cyc=1
_C.lambda_sty=1
_C.lambda_ds=1
_C.ds_iter=10000
_C.w_hpf=0
_C.randcrop_prob=0.5
_C.total_iters=10

_C.resume_iter=90000

_C.batch_size=1
_C.val_batch_size=8
_C.lr=1e-4
_C.f_lr=1e-6
_C.beta1=0.0
_C.beta2=0.99
_C.weight_decay=1e-4
_C.num_outs_per_domain=10
_C.mode='train'
_C.num_workers=4
_C.seed=777
_C.train_img_dir='data/train_continue'
_C.val_img_dir='data/val'
_C.sample_dir='expr/samples'
_C.checkpoint_dir='expr/checkpoints'
_C.eval_dir='expr/eval'
_C.result_dir='expr/results'
_C.src_dir='assets/src'
_C.ref_dir='assets/ref'
_C.inp_dir='assets/representative/custom/female'
_C.out_dir='assets/representative/celeba_hq/src/female'
_C.print_every=10
_C.sample_every=1000
_C.save_every=1000