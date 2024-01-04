

from pose_utils import load_colmap_data, save_poses

def sparse_to_posebounds(path):
    
    poses, pts3d, perm = load_colmap_data(path)
    save_poses(path, poses, pts3d, perm)
    
    print("saved in ", path)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/llff/flower_CT2/',
                        help='path to colmap sparse reconstruction')
    args = parser.parse_args()
    
    sparse_to_posebounds(args.path)
