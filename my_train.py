from arguments import GroupParams

class TrainDataset:
    sh_degree = 3
    # source_path = '/home/user/GauHuman/data/zju_mocap_refine/my_377'
    # model_path = './output/zju_mocap_refine/my_377_100_pose_correction_lbs_offset_split_clone_merge_prune'
    # exp_name = 'zju_mocap_refine/my_377_100_pose_correction_lbs_offset_split_clone_merge_prune'
    source_path = '/home/user/GauHuman/data/easy_mocap/'
    model_path = './output/easy_mocap/1_100_pose_correction_lbs_offset_split_clone_merge_prune'
    exp_name = 'easy_mocap/1_100_pose_correction_lbs_offset_split_clone_merge_prune'
    images = 'images'
    resolution = -1
    white_background = False
    data_device = 'cuda'
    eval = True
    smpl_type = 'smpl'
    actor_gender = 'neutral'
    motion_offset_flag = True

dataset = TrainDataset()

class OptConfig:
    iterations = 1200
    position_lr_init = 0.00016
    position_lr_final = 1.6e-06
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    pose_refine_lr = 5e-05
    lbs_offset_lr = 5e-05
    percent_dense = 0.01
    lambda_dssim = 0.2
    densification_interval = 100
    opacity_reset_interval = 3000
    densify_from_iter = 400
    densify_until_iter = 1000
    densify_grad_threshold = 0.0002


opt = OptConfig()


class Pipe:
    compute_cov3D_python = True
    convert_SHs_python = False
    debug = False


pipe = Pipe()

testing_iterations = [1200, 2000, 3000, 7000, 30000]
saving_iterations = [1200, 2000, 3000, 7000, 30000, 1200]
checkpoint_iterations = []
checkpoint = None
debug_from = -1


from scene.dataset_readers import CameraInfo, SceneInfo

    

if __name__ == '__main__':

    from train import training

    training(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        testing_iterations=testing_iterations,
        saving_iterations=saving_iterations,
        checkpoint_iterations=checkpoint_iterations,
        checkpoint=checkpoint,
        debug_from=debug_from,
    )

