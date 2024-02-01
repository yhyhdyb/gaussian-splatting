import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import mkdir_p
from plyfile import PlyData,PlyElement
import numpy as np


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        #2024.2.1       
        # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
        #                 np.asarray(plydata.elements[0]["y"]),
        #                 np.asarray(plydata.elements[0]["z"])),  axis=1)
        inputply = os.path.join(self.model_path, "input.ply")
        print(inputply)
        plydata=PlyData.read(inputply)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        x_min = np.min(positions[:, 0])
        x_max = np.max(positions[:, 0])
        y_min = np.min(positions[:, 1])
        y_max = np.max(positions[:, 1])
        z_min = np.min(positions[:, 2])
        z_max = np.max(positions[:, 2])
        #self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        path = os.path.join(point_cloud_path, "point_cloud.ply")
        mkdir_p(os.path.dirname(path))

        # Initialize an empty list to store the valid points
        valid_xyz = []
        valid_normals = []
        valid_f_dc = []
        valid_f_rest = []
        valid_opacities = []
        valid_scale = []
        valid_rotation = []

        xyz = self.gaussians._xyz.detach().cpu().numpy()   
        normals = np.zeros_like(xyz)
        f_dc = self.gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.gaussians._opacity.detach().cpu().numpy()
        scale = self.gaussians._scaling.detach().cpu().numpy()
        rotation = self.gaussians._rotation.detach().cpu().numpy()
        print(xyz.shape)
        # Iterate over the rows in xyz
        for i in range(xyz.shape[0]):
            # Get the x, y, z coordinates of the point
            x, y, z = xyz[i]
            # Check if the point is inside the bounding box
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                # If it is, add it and its corresponding attributes to the list of valid points
                valid_xyz.append([x, y, z])
                valid_normals.append(normals[i])
                valid_f_dc.append(f_dc[i])
                valid_f_rest.append(f_rest[i])
                valid_opacities.append(opacities[i])
                valid_scale.append(scale[i])
                valid_rotation.append(rotation[i])

        # Convert the lists back to numpy arrays
        xyz = np.array(valid_xyz)
        normals = np.array(valid_normals)
        f_dc = np.array(valid_f_dc)
        f_rest = np.array(valid_f_rest)
        opacities = np.array(valid_opacities)
        scale = np.array(valid_scale)
        rotation = np.array(valid_rotation)
        dtype_full = [(attribute, 'f4') for attribute in self.gaussians.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)




    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
