import os

from utils import read_extrinsics_binary, readColmapCameras, getNerfppNorm, storePly, fetchPly

path = "scenes/truck"

cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)


llffhold = 8

cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
cam_names = sorted(cam_names)
test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]

reading_dir = "images" if images == None else images
cam_infos_unsorted = readColmapCameras(
    cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
    images_folder=os.path.join(path, reading_dir), 
    depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
test_cam_infos = [c for c in cam_infos if c.is_test]

nerf_normalization = getNerfppNorm(train_cam_infos)

ply_path = os.path.join(path, "sparse/0/points3D.ply")
bin_path = os.path.join(path, "sparse/0/points3D.bin")
txt_path = os.path.join(path, "sparse/0/points3D.txt")
if not os.path.exists(ply_path):
    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    storePly(ply_path, xyz, rgb)
try:
    pcd = fetchPly(ply_path)
except:
    pcd = None