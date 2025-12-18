import numpy as np
import torch
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import load_images_and_cameras, load_images_and_cameras_processOGimages
import torch.nn.functional as F
from PIL import Image
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
        image_folder="example/nuscenesTmp1OG/images/",
        camera_folder="example/nuscenesTmp1OG/cameras_cam2ego/",  # Optional cameras_cam2world cameras_cam2ego
        # camera_folder=None,  # Optional
        depth_folder=None,   # Optional
        target_size=518,
        flag1v1=False,
    )

print("shape of images:", images.shape)           # (S, C, H, W) shape of images: torch.Size([3, 3, 518, 518])
# debug save first image
# resize to 252x448 for visualization
# img0 = images[0].permute(1,2,0).cpu().numpy()
# Image.fromarray((img0 * 255).astype(np.uint8)).save("debug_img0_og.png")
# img0 = np.array(F.interpolate(torch.from_numpy(img0).permute(2,0,1).unsqueeze(0), size=(252,448), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0))
# Image.fromarray((img0 * 255).astype(np.uint8)).save("debug_img0.png")


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



pose_enc = predictions['pose_enc']
extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:]) 
extrinsics_np = extrinsics[0].cpu().numpy()
intrinsics_np = intrinsics[0].cpu().numpy()
vggt_ext_world2cam = torch.from_numpy(extrinsics_np).to(device)
add_row = torch.tensor([0, 0, 0, 1], device=device).expand(vggt_ext_world2cam.size(0), 1, 4)
vggt_ext_world2cam = torch.cat((vggt_ext_world2cam, add_row), dim=1)
print("before normalized exts:\n", vggt_ext_world2cam)
# normalized_extrinsics = camera_normalization(vggt_ext_world2cam[0:1], vggt_ext_world2cam)
# print("after normalized exts:\n", normalized_extrinsics)
np.savez("./omnivggt_ext_1218_wPoseCam2Ego_518x294.npz", world2cam=vggt_ext_world2cam.cpu().numpy(), intrinsics=intrinsics_np)

# ========== 步骤1：确保depth是torch.Tensor（未转numpy前插值） ==========
depth_tensor = predictions['depth']  # 假设此时还是torch.Tensor，维度(S, V, H, W)
print(f"原始depth维度：{depth_tensor.shape}")  # 输出(S, V, H, W)，如(10,6,800,1280)

# ========== 步骤2：插值到448×448（双线性插值，保持S/V维度不变） ==========
# 1. 调整维度为(S*V, 1, H, W)：适配F.interpolate的输入格式（N,C,H,W）
N, C = depth_tensor.shape[0] * depth_tensor.shape[1], 1
H_ori, W_ori = depth_tensor.shape[2], depth_tensor.shape[3]
depth_reshaped = depth_tensor.reshape(N, C, H_ori, W_ori)


# 2. 双线性插值到448×448
depth_interp = F.interpolate(
    depth_reshaped, 
    size=(448, 448),  # (H_target, W_target)
    mode='bilinear',  # 深度图优先用双线性，避免近邻插值的块效应
    align_corners=False  # 避免边缘像素偏移，默认False更通用
)

# 3. 还原维度为(S, V, 448, 448)
depth_interp = depth_interp.reshape(depth_tensor.shape[0], depth_tensor.shape[1], 448, 448)
print(f"插值后depth维度：{depth_interp.shape}")  # 应输出(S, V, 448, 448)

# ========== 步骤3：转NumPy并保存 ==========
depth_np = depth_interp.cpu().numpy()
# np.savez("./omniggt_depthmaps_ego2cam_wGTpose.npz", depthmap=depth_np)

# 强制验证维度，避免索引错误
# print(f"depth_np维度详情：{depth_np.shape}")
for i in range(depth_np.shape[1]):
    # 用...代替:,:，适配任意分辨率
    depth_map = depth_np[0, i, ...]  
    print(f"View {i}: min={np.min(depth_map):.4f}, max={np.max(depth_map):.4f}")