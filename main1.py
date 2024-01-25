import numpy as np
import plyfile
import open3d
#import open3d as o3d
#pcd=open3d.io.read_point_cloud('points3D.bin')

# 读取points3D.ply文件
sparse_plydata = plyfile.PlyData.read('points3D.ply')
#sparse_plydata = plyfile.PlyData.read('points3D.bin')

# 获取稀疏点云数据
sparse_point_cloud = sparse_plydata['vertex'].data

# 计算包围盒的坐标
min_x = np.min(sparse_point_cloud['x'])
max_x = np.max(sparse_point_cloud['x'])
min_y = np.min(sparse_point_cloud['y'])
max_y = np.max(sparse_point_cloud['y'])
min_z = np.min(sparse_point_cloud['z'])
max_z = np.max(sparse_point_cloud['z'])

width=max_x-min_x
height=max_y-min_y
long=max_z-min_z

# 读取point_cloud.ply文件
gaussian_plydata = plyfile.PlyData.read('point_cloud.ply')

# 获取高斯场表示的点云数据
gaussian_point_cloud = gaussian_plydata['vertex'].data

# 裁剪点云数据
cropped_point_cloud = gaussian_point_cloud[(gaussian_point_cloud['x'] >= min_x) & (gaussian_point_cloud['x'] <= max_x) & (gaussian_point_cloud['y'] >= min_y) & (gaussian_point_cloud['y'] <= max_y) & (gaussian_point_cloud['z'] >= min_z) & (gaussian_point_cloud['z'] <= max_z)]

# 创建新的PlyData对象
cropped_plydata = plyfile.PlyData([plyfile.PlyElement.describe(cropped_point_cloud, 'vertex')])

# 保存为新的ply文件
cropped_plydata.write('cropped_point_cloud.ply')
print('finish')

# 裁剪点云数据 0.1
min_x = min_x+0.1*width
max_x = max_x-0.1*width
min_y = min_y+0.1*height
max_y = max_y-0.1*height
min_z = min_z+0.1*long
max_z = max_z-0.1*long

# 裁剪点云数据
cropped_point_cloud = gaussian_point_cloud[(gaussian_point_cloud['x'] >= min_x) & (gaussian_point_cloud['x'] <= max_x) & (gaussian_point_cloud['y'] >= min_y) & (gaussian_point_cloud['y'] <= max_y) & (gaussian_point_cloud['z'] >= min_z) & (gaussian_point_cloud['z'] <= max_z)]

# 创建新的PlyData对象
cropped_plydata = plyfile.PlyData([plyfile.PlyElement.describe(cropped_point_cloud, 'vertex')])

# 保存为新的ply文件
cropped_plydata.write('cropped_point_cloud_1.ply')
print('finish 1')

# 裁剪点云数据 0.2
alpha=0.2
min_x = min_x+alpha*width
max_x = max_x-alpha*width
min_y = min_y+alpha*height
max_y = max_y-alpha*height
min_z = min_z+0.0*long
max_z = max_z-0.0*long

# 裁剪点云数据
cropped_point_cloud = gaussian_point_cloud[(gaussian_point_cloud['x'] >= min_x) & (gaussian_point_cloud['x'] <= max_x) & (gaussian_point_cloud['y'] >= min_y) & (gaussian_point_cloud['y'] <= max_y) & (gaussian_point_cloud['z'] >= min_z) & (gaussian_point_cloud['z'] <= max_z)]

# 创建新的PlyData对象
cropped_plydata = plyfile.PlyData([plyfile.PlyElement.describe(cropped_point_cloud, 'vertex')])

# 保存为新的ply文件
cropped_plydata.write('cropped_point_cloud_2.ply')
print('finish 2')

# 裁剪点云数据 0.3
alpha=0.3
min_x = min_x+alpha*width
max_x = max_x-alpha*width
min_y = min_y+alpha*height
max_y = max_y-alpha*height
min_z = min_z+0.1*long
max_z = max_z-0.1*long

# 裁剪点云数据
cropped_point_cloud = gaussian_point_cloud[(gaussian_point_cloud['x'] >= min_x) & (gaussian_point_cloud['x'] <= max_x) & (gaussian_point_cloud['y'] >= min_y) & (gaussian_point_cloud['y'] <= max_y) & (gaussian_point_cloud['z'] >= min_z) & (gaussian_point_cloud['z'] <= max_z)]

# 创建新的PlyData对象
cropped_plydata = plyfile.PlyData([plyfile.PlyElement.describe(cropped_point_cloud, 'vertex')])

# 保存为新的ply文件
cropped_plydata.write('cropped_point_cloud_3.ply')
print('finish 3')
