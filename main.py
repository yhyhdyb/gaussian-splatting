# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2, mean observations per image: 2
# 1 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 P1180141.JPG
# 2362.39 248.498 58396 1784.7 268.254 59027 1784.7 268.254 -1
# 2 0.851773 0.0165051 0.503764 -0.142941 -0.737434 1.02973 3.74354 1 P1180142.JPG
# 1190.83 663.957 23056 1258.77 640.354 59070


# points3D.txt
# This file contains the information of all reconstructed 3D points in the dataset using one line per point, e.g.:

# # 3D point list with one line of data per point:
# #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# # Number of points: 3, mean track length: 3.3334
# 63390 1.67241 0.292931 0.609726 115 121 122 1.33927 16 6542 15 7345 6 6714 14 7227
# 63376 2.01848 0.108877 -0.0260841 102 209 250 1.73449 16 6519 15 7322 14 7212 8 3991
# 63371 1.71102 0.28566 0.53475 245 251 249 0.612829 118 4140 117 4473

import os
import collections
import numpy as np
import struct
import argparse


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def read_images_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_points3D_text(points3D, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# 定义一个函数，用于将points3D中的数据分组到n*n个瓦块中
# 参数points3D是一个字典，存储了三维点的id和属性
# 参数tiles是一个列表，存储了n*n个瓦块的x，y，z的最小值和最大值
def group_points_by_tiles(points3D, tiles):
    # 创建一个空字典，用于存储每个瓦块中的三维点
    groups = {}
    # 遍历每个瓦块
    for i, tile in enumerate(tiles):
        # 创建一个空列表，用于存储当前瓦块中的三维点
        group = []
        # 遍历points3D字典中的每个三维点
        for point3D_id, point3D in points3D.items():
            # 判断三维点是否在当前瓦块中
            if is_point_in_tile(point3D, tile):
                # 如果是，将三维点添加到当前瓦块的列表中
                group.append(point3D)
        # 将当前瓦块的列表存入字典中，以瓦块的索引作为键
        groups[i] = group
    # 返回分组后的字典
    return groups



# 定义一个函数，用于将每个瓦块中的三维点数据写入一个二进制文件
# 参数groups是一个字典，存储了每个瓦块中的三维点
# 参数path是模型文件的路径
def write_tiles_binary(groups, path):
    # 遍历每个瓦块
    for i, group in groups.items():
        # 生成每个瓦块对应的文件名，使用瓦块的索引作为后缀
        file_name = f"tile_{i}.bin"
        # 拼接文件的完整路径
        file_path = os.path.join(path, file_name)
        # 以二进制模式打开文件
        with open(file_path, "wb") as fid:
            # 写入当前瓦块中的三维点的数量，使用无符号长整型
            write_next_bytes(fid, len(group), "Q")
            # 遍历当前瓦块中的每个三维点
            for pt in group:
                # 写入三维点的id，使用无符号长整型
                write_next_bytes(fid, pt.id, "Q")
                # 写入三维点的坐标，使用双精度浮点型
                write_next_bytes(fid, pt.xyz.tolist(), "ddd")
                # 写入三维点的颜色，使用无符号字节型
                write_next_bytes(fid, pt.rgb.tolist(), "BBB")
                # 写入三维点的误差，使用双精度浮点型
                write_next_bytes(fid, pt.error, "d")
                # 获取三维点的轨迹长度
                track_length = pt.image_ids.shape[0]
                # 写入三维点的轨迹长度，使用无符号长整型
                write_next_bytes(fid, track_length, "Q")
                # 遍历三维点的轨迹元素
                for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                    # 写入图像id和二维点索引，使用有符号整型
                    write_next_bytes(fid, [image_id, point2D_id], "ii")

def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


# 定义一个函数，用于从二进制格式的模型文件中读取图像的数据
# 参数path_to_model_file是模型文件的路径
# 参考src/colmap/scene/reconstruction.cc中的Reconstruction类的ReadImagesBinary和WriteImagesBinary方法
def read_images_binary2(path_to_model_file):
    # 创建一个空字典，用于存储图像的数据
    images = {}
    # 以二进制模式打开模型文件
    with open(path_to_model_file, "rb") as fid:
        # 读取文件的前8个字节，解析为无符号长整型，得到图像的数量
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        # 遍历每个图像
        for _ in range(num_reg_images):
            # 读取文件的下64个字节，解析为无符号长整型、双精度浮点型和有符号整型，得到图像的id、姿态、相机id和名称长度
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            #fid, num_bytes=64, format_char_sequence="Qddddddi"
            # 获取图像的id
            image_id = binary_image_properties[0]
            # 获取图像的姿态，转换为numpy数组
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            # 获取图像的相机id
            #camera_id = binary_image_properties[7]
            camera_id = binary_image_properties[8]
            # 获取图像的名称长度
            image_name_len = binary_image_properties[9]
            # 读取文件的下image_name_len个字节，解析为字符串，得到图像的名称
            image_name = "".join(
                read_next_bytes(fid, image_name_len, "c").astype(str)
            )
            # 读取文件的下8个字节，解析为无符号长整型，得到图像的二维点的数量
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            # 读取文件的下24*num_points2D个字节，解析为双精度浮点型和无符号长整型，得到图像的二维点的坐标和三维点的id
            binary_point2D_properties = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddQ" * num_points2D,
            )
            # 获取图像的二维点的坐标，转换为numpy数组
            xys = np.array(tuple(map(float, binary_point2D_properties[0::3])))
            xys = xys.reshape((num_points2D, 2))
            # 获取图像的二维点对应的三维点的id，转换为numpy数组
            point3D_ids = np.array(tuple(map(int, binary_point2D_properties[2::3])))
            # 将图像的id和属性封装为Image对象，存入字典中
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    # 返回图像的数据
    return images



'''
def is_point_in_tile(point, tile):
    # 获取三维点的x，y，z坐标
    x = point.xyz[0]
    y = point.xyz[1]
    z = point.xyz[2]
    # 获取瓦块的x，y，z的最小值和最大值
    x_min = tile[0]
    x_max = tile[1]
    y_min = tile[2]
    y_max = tile[3]
    z_min = tile[4]
    z_max = tile[5]
    # 判断三维点的x，y，z坐标是否在瓦块的范围内
    if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
        # 如果是，返回True
        return True
    else:
        # 如果不是，返回False
        return False
'''

def is_point_in_tile(point, tile):
    # 获取三维点的坐标
    x, y, z = point.xyz
    # 获取瓦块的x，y，z的最小值和最大值
    x_min, x_max, y_min, y_max, z_min, z_max = tile
    # 判断三维点是否在瓦块的x，y，z的范围内
    return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max


# 定义一个函数，用于判断一个图像是否在一个瓦块中
# 参数image是一个Image对象，存储了图像的id和属性
# 参数tile是一个列表，存储了一个瓦块的x，y，z的最小值和最大值
# 参数points3D是一个字典，存储了三维点的id和属性
def is_image_in_tile(image, tile, points3D):
    # 获取图像的二维点对应的三维点的id
    point3D_ids = image.point3D_ids
    # 遍历每个三维点的id
    for point3D_id in point3D_ids:
        # 如果三维点的id不是-1，表示该二维点有对应的三维点
        if point3D_id != -1:
              # 检查point3D_id是否在points3D字典中
            if point3D_id in points3D.keys():
                # 从points3D字典中获取该三维点的对象
                print(6666666666666666666666666666666666666666666666666666666666)
                point3D = points3D[point3D_id]
                # 判断该三维点是否在当前瓦块中
                if is_point_in_tile(point3D, tile):
                    # 如果是，返回True，表示该图像在当前瓦块中
                    return True
            else:
                print(f"Warning: point3D_id {point3D_id} not found in points3D")
            # # 从points3D字典中获取该三维点的对象
            # print(point3D_id)
            # point3D = points3D[point3D_id]
            # # 判断该三维点是否在当前瓦块中
            # if is_point_in_tile(point3D, tile):
            #     # 如果是，返回True，表示该图像在当前瓦块中
            #     return True
    # 如果遍历完所有的三维点，都没有找到在当前瓦块中的，返回False，表示该图像不在当前瓦块中
    return False



# 定义一个函数，用于将images中的数据分组到n*n个瓦块中
# 参数images是一个字典，存储了图像的id和属性
# 参数tiles是一个列表，存储了n*n个瓦块的x，y，z的最小值和最大值
# 参数points3D是一个字典，存储了三维点的id和属性
def group_images_by_tiles(images, tiles, points3D):
    # 创建一个空字典，用于存储每个瓦块中的图像
    groups = {}
    # 遍历每个瓦块
    for i, tile in enumerate(tiles):
        # 创建一个空列表，用于存储当前瓦块中的图像
        group = []
        # 遍历images字典中的每个图像
        for image in images.values():
            # 判断图像是否在当前瓦块中
            if is_image_in_tile(image, tile, points3D):
                # 如果是，将图像添加到当前瓦块的列表中
                group.append(image)
        # 将当前瓦块的列表存入字典中，以瓦块的索引作为键
        groups[i] = group
    # 返回分组后的字典
    return groups
# 定义一个函数，用于将每个瓦块中的图像数据写入一个二进制文件
# 参数groups是一个字典，存储了每个瓦块中的图像
# 参数path是模型文件的路径


# 定义一个函数，用于判断一个相机是否在一个瓦块中
def is_camera_in_tile(camera, tile,groups_images,points3D,tile_id):
    camera_id = camera.id   
    for image in groups_images[tile_id]:
        if image.camera_id == camera_id and is_image_in_tile(image, tile, points3D):
            return True
    return False



# group_images:dict
def group_cameras_by_tiles(cameras, tiles, groups_images, points3D):
    # 创建一个空字典，用于存储每个瓦块中的相机
    groups = {}
    # 遍历每个瓦块
    for idx, tile in enumerate(tiles):
        # 创建一个空列表，用于存储当前瓦块中的相机
        group = []
        # 遍历cameras字典中的每个相机
        for camera in cameras.values():
            # 判断相机是否在当前瓦块中
            if is_camera_in_tile(camera, tile, groups_images,points3D,idx):
                #i是瓦块的索引
                # 如果是，将相机添加到当前瓦块的列表中
                group.append(camera)
        # 将当前瓦块的列表存入字典中，以瓦块的索引作为键
        groups[idx] = group
    # 返回分组后的字典
    return groups

def write_tiles_cameras_binary(groups, path):
    # 遍历每个瓦块
    for i, group in groups.items():
        # 生成每个瓦块对应的文件名，使用瓦块的索引作为后缀
        file_name = f"tile_{i}.bin"
        # 拼接文件的完整路径
        file_path = os.path.join(path, file_name)
        # 以二进制模式打开文件
        with open(file_path, "wb") as fid:
            write_next_bytes(fid, len(group), "Q")
            for cam in group:
                model_id = CAMERA_MODEL_NAMES[cam.model].model_id
                camera_properties = [cam.id, model_id, cam.width, cam.height]
                write_next_bytes(fid, camera_properties, "iiQQ")
                for p in cam.params:
                    write_next_bytes(fid, float(p), "d")




def write_tiles_images_binary(groups, path):
    # 遍历每个瓦块
    for i, group in groups.items():
        # 生成每个瓦块对应的文件名，使用瓦块的索引作为后缀
        file_name = f"tile_{i}.bin"
        # 拼接文件的完整路径
        file_path = os.path.join(path, file_name)
        # 以二进制模式打开文件
        with open(file_path, "wb") as fid:
            # 写入当前瓦块中的图像的数量，使用无符号长整型
            write_next_bytes(fid, len(group), "Q")
            # 遍历当前瓦块中的每个图像
            #for image in group:
            for image in group:
                # 写入图像的id，使用无符号长整型
                write_next_bytes(fid, image.id, "i")
                #write_next_bytes(fid, image.id, "Q")
                # 写入图像的姿态，使用双精度浮点型
                write_next_bytes(fid, image.qvec.tolist(), "dddd")
                write_next_bytes(fid, image.tvec.tolist(), "ddd")
                # 写入图像的相机id，使用无符号长整型
                write_next_bytes(fid, image.camera_id, "i")
                for char in image.name:
                    write_next_bytes(fid, char.encode("utf-8"), "c")
                write_next_bytes(fid, b"\x00", "c")
                write_next_bytes(fid, len(image.point3D_ids), "Q")
                for xy, p3d_id in zip(image.xys, image.point3D_ids):
                    write_next_bytes(fid, [*xy, p3d_id], "ddq")

def main():
# c: char (a single character)
# e: half precision float
# f: float
# d: double
# h: short
# H: unsigned short
# i: int
# I: unsigned int
# l: long
# L: unsigned long
# q: long long
# Q: unsigned long long

    # path_to_input_point = r"/home/kanjing/3Dconstruction/gs24/data/xiaobieshu/sparse/0/points3D.bin"
    # path_to_image_file = r"/home/kanjing/3Dconstruction/gs24/data/xiaobieshu/sparse/0/images.bin"
    # path_to_camera_file = r"/home/kanjing/3Dconstruction/gs24/data/xiaobieshu/sparse/0/cameras.bin"

    # path_to_pointout = r"/home/kanjing/3Dconstruction/gs24/data/tile/xiaobieshu/point3Dout"
    # path_to_imgout = r"/home/kanjing/3Dconstruction/gs24/data/tile/xiaobieshu/imagesout"
    # path_to_cameraout = r"/home/kanjing/3Dconstruction/gs24/data/tile/xiaobieshu/camerasout"

    path_to_input_point = r"/home/kanjing/3Dconstruction/gs24/data/garden/sparse/0/points3D.bin"
    path_to_image_file = r"/home/kanjing/3Dconstruction/gs24/data/garden/sparse/0/images.bin"
    path_to_camera_file = r"/home/kanjing/3Dconstruction/gs24/data/garden/sparse/0/cameras.bin"

    path_to_pointout = r"/home/kanjing/3Dconstruction/gs24/data/tile/garden/point3Dout"
    path_to_imgout = r"/home/kanjing/3Dconstruction/gs24/data/tile/garden/imagesout"
    path_to_cameraout = r"/home/kanjing/3Dconstruction/gs24/data/tile/garden/camerasout"
    points3D = {}
    with open(path_to_input_point, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        print(num_points)
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )

            # 创建一个空列表，用于存储所有三维点的xyz
            xyz_list = []
            # 遍历points3D字典中的每个三维点
            for point3D_id, point3D in points3D.items():
                # 将三维点的xyz添加到列表中
                xyz_list.append(point3D.xyz)
            # 将列表转换为numpy数组
            xyz_array = np.array(xyz_list)
            # 计算xyz数组中的x，y，z的最小值和最大值
            x_min = np.min(xyz_array[:, 0])
            x_max = np.max(xyz_array[:, 0])
            y_min = np.min(xyz_array[:, 1])
            y_max = np.max(xyz_array[:, 1])
            z_min = np.min(xyz_array[:, 2])
            z_max = np.max(xyz_array[:, 2])
            # 打印结果
            '''
            print(f"x的最小值是{x_min}")
            print(f"x的最大值是{x_max}")
            print(f"y的最小值是{y_min}")
            print(f"y的最大值是{y_max}")
            print(f"z的最小值是{z_min}")
            print(f"z的最大值是{z_max}")
            '''
            tiles = []
            n=2
            overlap=0.15
            # 计算包围盒的x，y，z的范围
            bbox = [-1.0, 1.0, -2.0, 2.0, -3.0, 3.0]
            bbox[0]=x_min
            bbox[1]=x_max
            bbox[2]=y_min
            bbox[3]=y_max
            bbox[4]=z_min
            bbox[5]=z_max
            '''
            bbox[0] = 0
            bbox[1] = 200
            bbox[2] = 0
            bbox[3] = 200
            bbox[4] = 0
            bbox[5] = 200
            '''
            x_range = bbox[1] - bbox[0]
            y_range = bbox[3] - bbox[2]
            z_range = bbox[5] - bbox[4]
            # 计算瓦块的x，y，z的范围，考虑重合度
            tile_x_range = x_range / (n - (n - 1) * overlap)
            tile_y_range = y_range / (n - (n - 1) * overlap)
            tile_z_range = z_range
            # 遍历每一行
            for i in range(n):
                # 计算当前行的y的最小值和最大值
                y_min = bbox[2] + i * tile_y_range * (1 - overlap)
                y_max = y_min + tile_y_range
                # 遍历每一列
                for j in range(n):
                    # 计算当前列的x的最小值和最大值
                    x_min = bbox[0] + j * tile_x_range * (1 - overlap)
                    x_max = x_min + tile_x_range
                    # 计算当前层的z的最小值和最大值，与包围盒相同
                    z_min = bbox[4]
                    z_max = bbox[5]
                    # 将当前瓦块的x，y，z的最小值和最大值添加到列表中
                    tiles.append([x_min, x_max, y_min, y_max, z_min, z_max])
    


    groups = {}
    groups = group_points_by_tiles(points3D, tiles)
    write_tiles_binary(groups, path_to_pointout)
    images = read_images_binary(path_to_image_file)
    groups_images = {}
    groups_images = group_images_by_tiles( images, tiles, points3D)
    write_tiles_images_binary(groups_images, path_to_imgout)
    groups_cameras = {}
    cameras = read_cameras_binary(path_to_camera_file)
    groups_cameras = group_cameras_by_tiles(cameras, tiles, groups_images,points3D)
    write_tiles_cameras_binary(groups_cameras, path_to_cameraout)


def main2():
    parser = argparse.ArgumentParser(
        description="Read and write COLMAP binary and text models"
    )
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument(
        "--input_format",
        choices=[".bin", ".txt"],
        help="input model format",
        default="",
    )
    parser.add_argument("--output_model", help="path to output model folder")
    parser.add_argument(
        "--output_format",
        choices=[".bin", ".txt"],
        help="outut model format",
        default=".txt",
    )
    args = parser.parse_args()

    cameras, images, points3D = read_model(
        path=args.input_model, ext=args.input_format
    )

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))

    if args.output_model is not None:
        write_model(
            cameras,
            images,
            points3D,
            path=args.output_model,
            ext=args.output_format,
        )


if __name__ == "__main__":
    main()
