import numpy as np
import torch
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import load_images_and_cameras, load_images_and_cameras_processOGimages
from omnivggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
import torch.nn.functional as F
from PIL import Image
import scipy.interpolate
from pathlib import Path
def world_coords_points_to_color_image(
    points: torch.Tensor,
    colors: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    size: tuple,
) -> torch.Tensor:
    """
    将带有颜色的3D世界坐标点投影到相机视角，生成一张背景为白色的稀疏彩色图像。
    
    Args:
        points (torch.Tensor): 世界坐标系下的点云 (N, 3)。
        colors (torch.Tensor): 点云对应的颜色 (N, 3)，范围 0-1。
        extrinsic (torch.Tensor): 相机外参矩阵 (4, 4) 或 (3, 4)。
        intrinsic (torch.Tensor): 相机内参矩阵 (3, 3)。
        size (tuple): 输出图像的尺寸 (height, width)。
        
    Returns:
        torch.Tensor: 渲染出的稀疏彩色图像 (height, width, 3)，范围 0-1。
    """

    height, width = size
    device = points.device
    points = points.to(torch.float32).to(device)
    colors = colors.to(torch.float32).to(device)
    extrinsic = extrinsic.to(torch.float32).to(device)
    intrinsic = intrinsic.to(torch.float32).to(device)
    # 1. 坐标系转换 (世界 -> 相机)
    if extrinsic.shape == (3, 4):
        extrinsic_4x4 = torch.eye(4, device=device)
        extrinsic_4x4[:3, :] = extrinsic
    else:
        extrinsic_4x4 = extrinsic

    R = extrinsic_4x4[:3, :3]
    T = extrinsic_4x4[:3, 3:]
    cam_points = points @ R.T + T.T

    # 2. 投影
    valid_depth_mask = cam_points[:, 2] > 1e-4
    if not valid_depth_mask.any():
        return torch.ones((height, width, 3), device=device) # 返回纯白图像

    cam_points = cam_points[valid_depth_mask]
    points_colors = colors[valid_depth_mask]
    
    projected_points = (intrinsic @ cam_points.T).T
    pixel_coords = projected_points[:, :2] / projected_points[:, 2:3]

    # --- FIX: 先四舍五入，再进行边界检查 ---
    # 计算整数像素坐标
    u_all = torch.round(pixel_coords[:, 0]).to(torch.long)
    v_all = torch.round(pixel_coords[:, 1]).to(torch.long)
    
    # 在整数坐标上进行边界检查
    valid_pixel_mask = (u_all >= 0) & (u_all < width) & (v_all >= 0) & (v_all < height)
    if not valid_pixel_mask.any():
        return torch.ones((height, width, 3), device=device, dtype=torch.float32)

    # 使用掩码筛选出所有有效的数据
    u_valid = u_all[valid_pixel_mask]
    v_valid = v_all[valid_pixel_mask]
    depths_valid = cam_points[valid_pixel_mask, 2]
    colors_valid = points_colors[valid_pixel_mask]
    # --- END FIX ---

    # 初始化为白色背景
    color_map = torch.ones((height, width, 3), device=device)
    
    # Z-buffering: 从远到近排序以正确处理遮挡
    sorted_indices = torch.argsort(depths_valid, descending=True)
    u_sorted = u_valid[sorted_indices]
    v_sorted = v_valid[sorted_indices]
    colors_sorted = colors_valid[sorted_indices]
    
    color_map[v_sorted, u_sorted] = colors_sorted
            
    return color_map

def interpolate_sparse_image(sparse_image_np: np.ndarray) -> np.ndarray:
    """
    使用scipy.interpolate.griddata对稀疏图像进行插值填充。
    
    Args:
        sparse_image_np (np.ndarray): (H, W, 3) 的NumPy数组，背景为白色(1.0)，
                                     前景为稀疏的彩色点。
                                     
    Returns:
        np.ndarray: 插值后的密集图像 (H, W, 3)。
    """
    height, width, _ = sparse_image_np.shape
    
    # 找出已知像素点（非背景色）的坐标和颜色值
    # 由于浮点数精度问题，我们检查是否“不是”纯白色
    is_known = (sparse_image_np.sum(axis=2) < 2.999)
    known_points_coords = np.argwhere(is_known) # (num_known, 2) -> (y, x)
    known_points_values = sparse_image_np[is_known] # (num_known, 3)
    
    if len(known_points_coords) < 3:
        # 如果已知点太少，无法进行有意义的插值，直接返回原图
        print("警告: 已知点过少，跳过插值。")
        return sparse_image_np

    # 创建我们想要插值的目标网格坐标
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    
    # 使用griddata进行插值
    # 我们需要对每个颜色通道独立进行
    print("开始进行插值...")
    interpolated_image = scipy.interpolate.griddata(
        points=known_points_coords,
        values=known_points_values,
        xi=(grid_y, grid_x),
        method='linear', # 'linear'是速度和效果的良好折中
        fill_value=1.0 # 未定义区域（凸包外）填充白色
    )
    print("插值完成。")
    
    # 确保值在[0, 1]范围内
    interpolated_image = np.clip(interpolated_image, 0, 1)
    
    return interpolated_image

def modify_intrinsics_for_new_width(intrinsic, old_width, new_width):
    """根据新的宽度调整内参矩阵。"""
    new_intrinsic = intrinsic.clone()
    new_intrinsic[..., 0, 2] = new_intrinsic[..., 0, 2] * (new_width / old_width)
    return new_intrinsic

def main_reprojection_pipeline(
    world_coords_points_map,
    images,
    extrinsics,
    intrinsics,
    image_names,
    new_width
):
    """
    主流程函数：合并点云，重投影，插值并保存。
    """
    device = world_coords_points_map.device
    num_views, C, H, W = images.shape
    new_height = H
    
    all_points = world_coords_points_map.reshape(-1, 3)
    all_colors = images.permute(0, 2, 3, 1).reshape(-1, 3)
    print(f"成功合并点云，总点数: {all_points.shape[0]}")

    modified_intrinsics = []
    for i in range(num_views):
        original_i = intrinsics[0, i]
        modified_i = modify_intrinsics_for_new_width(original_i, W, new_width)
        modified_intrinsics.append(modified_i)

    for i in range(num_views):
        print(f"\n--- 正在为视角 {i+1}/{num_views} 进行重投影 ---")
        
        extrinsic_i = extrinsics[0, i]
        intrinsic_i_mod = modified_intrinsics[i]
        
        # 1. 渲染稀疏彩色图像（白色背景）
        reprojected_sparse_tensor = world_coords_points_to_color_image(
            points=all_points,
            colors=all_colors,
            extrinsic=extrinsic_i,
            intrinsic=intrinsic_i_mod,
            size=(new_height, new_width)
        )
        
        # 准备保存路径
        if isinstance(image_names, str):
            # 如果输入的是字符串（如"801"），保存在 outputsImages/801/ 下，并以索引为文件名
            output_dir = Path("outputsImages") / image_names
            base_name = str(i)
        else:
            original_path = Path(image_names[i])
            stem = original_path.stem
            base_name = stem.rsplit('_', 1)[0]
            output_dir = Path(str(original_path.parent).replace("examples", "unprojection_outputs"))
            
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. 保存稀疏图像
        reprojected_sparse_np = (reprojected_sparse_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image_sparse = Image.fromarray(reprojected_sparse_np)
        
        output_filename_sparse = f"{base_name}_reprojected_{new_height}x{new_width}_color.png"
        output_path_sparse = output_dir / output_filename_sparse
        pil_image_sparse.save(output_path_sparse)
        print(f"稀疏图像已保存至: {output_path_sparse}")
        
        # 3. 对稀疏图像进行插值
        # 将Tensor (0-1范围) 转为Numpy数组进行插值
        interpolated_np = interpolate_sparse_image(reprojected_sparse_tensor.cpu().numpy())
        
        # 4. 保存插值后的图像
        interpolated_np_uint8 = (interpolated_np * 255).astype(np.uint8)
        pil_image_interpolated = Image.fromarray(interpolated_np_uint8)
        
        output_filename_inter = f"{base_name}_reprojected_{new_height}x{new_width}_color_inter.png"
        output_path_inter = output_dir / output_filename_inter
        pil_image_interpolated.save(output_path_inter)
        print(f"插值图像已保存至: {output_path_inter}")

def camera_normalization(pivotal_pose: torch.Tensor, poses: torch.Tensor):
    # [1, 4, 4], [N, 4, 4]
    
    canonical_camera_extrinsics = torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]], dtype=torch.float32, device=pivotal_pose.device)
    pivotal_pose_inv = torch.inverse(pivotal_pose)
    camera_norm_matrix = torch.bmm(canonical_camera_extrinsics, pivotal_pose_inv)
    
    # normalize all views
    poses = torch.bmm(camera_norm_matrix.repeat(poses.shape[0], 1, 1), poses)

    return poses


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = OmniVGGT().to(device)
from safetensors.torch import load_file
state_dict = load_file("./checkpoints/OmniVGGT.safetensors")
model.load_state_dict(state_dict, strict=True)
model.eval()

# Load and preprocess images
# images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices = \
#     load_images_and_cameras(
#         image_folder="example/nuscenesTmp1/images/",
#         # camera_folder="example/nuscenesTmp1/cameras_cam2world/",  # Optional
#         camera_folder=None,  # Optional
#         depth_folder=None,   # Optional
#         target_size=448,
#         flag1v1=False,
#     )
images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices =  load_images_and_cameras_processOGimages(
        image_folder="example/nuscenesTmp1OG/037020f/",
        camera_folder="example/nuscenesTmp1OG/cameras_cam2world/",  # Optional cameras_cam2world cameras_cam2ego
        # camera_folder=None,  # Optional
        depth_folder=None,   # Optional
        target_size=518,
        flag1v1=False,
    )

print("shape of images:", images.shape)           # (S, C, H, W) shape of images: torch.Size([3, 3, 294, 518])

# --- Camera Normalization ---
# 将相机外参归一化，使第一帧相机位于世界坐标系原点 (Identity)
print("Normalizing camera poses relative to the first frame...")
S = extrinsics.shape[1]
# 1. 补齐为 4x4 矩阵并转到 C2W (Camera-to-World)
extrinsics_4x4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, S, 1, 1)
extrinsics_4x4[:, :, :3, :4] = extrinsics.to(device)
c2w = torch.inverse(extrinsics_4x4[0]) # (S, 4, 4)

# 2. 调用已有的 camera_normalization 函数进行归一化
normalized_c2w = camera_normalization(c2w[0:1], c2w) # 返回 (S, 4, 4)

# 3. 转回 W2C (World-to-Camera) 并裁剪回 3x4
normalized_w2c = torch.inverse(normalized_c2w)
extrinsics = normalized_w2c.unsqueeze(0)[:, :, :3, :4]
# ----------------------------


# Prepare inputs
inputs = {
    'images': images.to(device),
    'extrinsics': extrinsics.to(device),
    'intrinsics': intrinsics.to(device),
    'depth': depthmaps.to(device),
    'mask': masks.to(device),
    'depth_gt_index': depth_indices,
    'camera_gt_index': camera_indices
}

# Run inference
with torch.no_grad():
    predictions = model(**inputs)


# Use GT poses instead of predicted ones
print("Using GT poses for unprojection and reprojection.")
extrinsics = inputs['extrinsics']
intrinsics = inputs['intrinsics']
print("extrinsics shape (GT):", extrinsics)  # (1, S, 4, 4)
# pose_enc = predictions['pose_enc']
# extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:]) 
extrinsics_np = extrinsics[0].cpu().numpy()
intrinsics_np = intrinsics[0].cpu().numpy()
vggt_ext_world2cam = torch.from_numpy(extrinsics_np).to(device)
add_row = torch.tensor([0, 0, 0, 1], device=device).expand(vggt_ext_world2cam.size(0), 1, 4)
vggt_ext_world2cam = torch.cat((vggt_ext_world2cam, add_row), dim=1)
# print("before normalized exts:\n", vggt_ext_world2cam)


# ========== 步骤1：确保depth是torch.Tensor（未转numpy前插值） ==========
depth_tensor = predictions['depth']  # 假设此时还是torch.Tensor，维度(S, V, H, W)
print(f"原始depth维度：{depth_tensor.shape}")  # 原始depth维度：torch.Size([1, 3, 294, 518, 1])
print("extrinsics shape:", extrinsics.shape)  
print("intrinsics shape:", intrinsics.shape)

world_coords_points_map_by_unprojection = unproject_depth_map_to_point_map(depth_tensor.squeeze(0), 
                                                            extrinsics.squeeze(0), 
                                                            intrinsics.squeeze(0))
world_coords_points_map_by_unprojection = torch.from_numpy(world_coords_points_map_by_unprojection)
# print(world_coords_points_map_by_unprojection.shape) # (3, 280, 518, 3) 
image_names = "801_1gtpose"     
main_reprojection_pipeline(
    world_coords_points_map_by_unprojection,
    images,
    extrinsics,
    intrinsics,
    image_names,
    new_width=1036
)

# python tmpRun_nuScenesUseGTpose.py