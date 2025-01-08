import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    # experiment
    parser.add_argument('--load-image', type=str, default=None,
                        help='zero123pp image path')
    parser.add_argument("--proj-name", type=str, default="test",
                        help='experiment name')
    parser.add_argument("--wandb-project", type=str, 
                        default="zerorf", help='wandb project name')
    
    # data
    parser.add_argument("--dataset", type=str, 
                        default="nerf_syn", help='type of dataset')
    parser.add_argument("--data-dir", type=str, 
                        default="/root/nerf_synthetic", help='directory of the dataset')
    parser.add_argument("--obj", type=str, 
                        default="chair", help='object name')
    parser.add_argument("--n-views", type=int, 
                        default=4, help='number of input views')
    
    # model
    parser.add_argument("--model-res", type=int, 
                        default=20, help='noise resolution (should be about 1/40 the provided image resolution), ignored when load-image is set')
    parser.add_argument("--model-ch", type=int, 
                        default=8, help='noise channel')
    parser.add_argument("--n-rays-init", type=int, 
                        default=2**12, help='number of rays per batch initially')
    parser.add_argument("--n-rays-up", type=int, 
                        default=2**16, help='number of rays per batch after 100 iterations')
    parser.add_argument("--learn-bg", action='store_true', help='if learn background')
    parser.add_argument("--bg-color", type=float, 
                        default=1.0, help='background color')
    parser.add_argument("--rep", type=str, choices=['dif', 'tensorf'],
                        default="dif", help="representation to use")
    
    # training
    parser.add_argument("--net-lr", type=float, 
                        default=0.002, help='learning rate')
    parser.add_argument("--seed", type=int, 
                        default=1337, help='random seed')
    parser.add_argument("--n-val", type=int, 
                        default=1, help='number of validate views')
    parser.add_argument("--net-lr-decay-to", type=float, 
                        default=0.002, help='lr decay rate')
    parser.add_argument("--n-iters", type=int, 
                        default=10000, help='number of iterations')
    parser.add_argument("--val-iter", type=int, 
                        default=1000, help='valid every k iterations')
    parser.add_argument("--device", type=str, 
                        default="cuda:0", help='device name')
    
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()