import os
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

# ---------- Image Utility ----------

"""

 -> 6-images

"""

def image_list(images):
    assert len(images) == 6

    image_np_array = []
    for image in images:
        image_np_array.append(np.array(image))

    concatenated_image = np.concatenate(image_np_array, axis=1)


    # horizontal to dice
    assert concatenated_image.shape[0] * 6 == concatenated_image.shape[1]

    w = concatenated_image.shape[0]
    cube_dice = np.zeros((w * 3, w * 4, concatenated_image.shape[2]), dtype=concatenated_image.dtype)
    # cube_list = cube_h2list(concatenated_image)

    cube_list = np.split(concatenated_image, 6, axis=1)

    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_list[i]
        if i in [1, 2]:
            face = np.flip(face, axis=1)
        if i == 4:
            face = np.flip(face, axis=0)
        cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w] = face

    
    # dice to horizontal

    w = cube_dice.shape[0] // 3
    assert cube_dice.shape[0] == w * 3 and cube_dice.shape[1] == w * 4
    cube_h = np.zeros((w, w * 6, cube_dice.shape[2]), dtype=cube_dice.dtype)
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[sy*w:(sy+1)*w, sx*w:(sx+1)*w]
        # if i in [1, 2]:
        #     face = np.flip(face, axis=1)
        # if i == 4:
        #     face = np.flip(face, axis=0)
        cube_h[:, i*w:(i+1)*w] = face
    
    
    fixed_horizontal_images = cube_h

    return fixed_horizontal_images


# -------------- Main Logic --------------

"""

generates the grid of u,v coordinates from 
u = -pi to pi 
v = pi/2 to -pi/2

Mathematical notation. You can relate it with Longitude and Latitude

u ∈ [−π,π] ,

v ∈ [ π/2, - π/2]

"""

def equirect_uvgrid(h, w):
    u = np.linspace(-np.pi, np.pi, num=w, dtype=np.float32)
    v = np.linspace(np.pi, -np.pi, num=h, dtype=np.float32) / 2

    return np.stack(np.meshgrid(u, v), axis=-1)

"""

creates a mapping for cube images

such that it represents which cube map maps to which equirectangle projection.
also determines the boundaries

"""

def equirect_facetype(h, w):
    '''
    0F 1R 2B 3L 4U 5D
    '''
    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1

    """
    {x, y, z}

     -X        +x
    {-------|-------}

    """

    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)

"""

applies the transformation to spherical coordinates, we acquired previously.
The padding is to handle the edges.

"""

def sample_cubefaces(cube_faces, tp, coor_y, coor_x, order):
    cube_faces = cube_faces.copy()
    cube_faces[1] = np.flip(cube_faces[1], 1)
    cube_faces[2] = np.flip(cube_faces[2], 1)
    cube_faces[4] = np.flip(cube_faces[4], 0)

    # Pad up down
    pad_ud = np.zeros((6, 2, cube_faces.shape[2]))
    pad_ud[0, 0] = cube_faces[5, 0, :]
    pad_ud[0, 1] = cube_faces[4, -1, :]
    pad_ud[1, 0] = cube_faces[5, :, -1]
    pad_ud[1, 1] = cube_faces[4, ::-1, -1]
    pad_ud[2, 0] = cube_faces[5, -1, ::-1]
    pad_ud[2, 1] = cube_faces[4, 0, ::-1]
    pad_ud[3, 0] = cube_faces[5, ::-1, 0]
    pad_ud[3, 1] = cube_faces[4, :, 0]
    pad_ud[4, 0] = cube_faces[0, 0, :]
    pad_ud[4, 1] = cube_faces[2, 0, ::-1]
    pad_ud[5, 0] = cube_faces[2, -1, ::-1]
    pad_ud[5, 1] = cube_faces[0, -1, :]
    cube_faces = np.concatenate([cube_faces, pad_ud], 1)

    # Pad left right
    pad_lr = np.zeros((6, cube_faces.shape[1], 2))
    pad_lr[0, :, 0] = cube_faces[1, :, 0]
    pad_lr[0, :, 1] = cube_faces[3, :, -1]
    pad_lr[1, :, 0] = cube_faces[2, :, 0]
    pad_lr[1, :, 1] = cube_faces[0, :, -1]
    pad_lr[2, :, 0] = cube_faces[3, :, 0]
    pad_lr[2, :, 1] = cube_faces[1, :, -1]
    pad_lr[3, :, 0] = cube_faces[0, :, 0]
    pad_lr[3, :, 1] = cube_faces[2, :, -1]
    pad_lr[4, 1:-1, 0] = cube_faces[1, 0, ::-1]
    pad_lr[4, 1:-1, 1] = cube_faces[3, 0, :]
    pad_lr[5, 1:-1, 0] = cube_faces[1, -2, :]
    pad_lr[5, 1:-1, 1] = cube_faces[3, -2, ::-1]
    cube_faces = np.concatenate([cube_faces, pad_lr], 2)

    return map_coordinates(cube_faces, [tp, coor_y, coor_x], order=order, mode='wrap')


"""

Main Function to convert the cube maps to equirectangle image

- Calculates the U,V with helper function
- Creates Mapping
- Normalize the coordinates
- Reconstruction of image

"""

def cube_to_equirectanlge(cubemap, h, w, mode='bilinear'):
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')
    
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    assert w % 8 == 0
    face_w = cubemap.shape[0]

    uv = equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack([
        sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
        for i in range(cube_faces.shape[3])
    ], axis=-1)

    return equirec


# ------- Load Images ------

fixed_image = image_list(
    [
        Image.open('front.jpeg'),  # F
        Image.open('right.jpeg'),  # R
        Image.open('back.jpeg'),   # B
        Image.open('left.jpeg'),   # L
        Image.open('up.jpeg'),     # U
        Image.open('down.jpeg')    # D
     ]
)

pano_image = cube_to_equirectanlge(fixed_image, 1024, 1280, mode='nearest')

# img = Image.fromarray(pano_image.astype(np.uint8), 'RGB')

# img = Image.fromarray(fixed_image, 'RGB')
# img.show()


def generate_folder_info(root_folder):
    folder_info = []

    for subdir, _, files in os.walk(root_folder):
        image_files = [f for f in files if f.lower().endswith('jpg')]

        if len(image_files) == 6:
            relative_path = os.path.relpath(subdir, root_folder)
            parts = relative_path.split(os.sep)

            # print("parts ", parts)

            if len(parts) > 2:
                level1 = parts[0]
                level2 = parts[1]
                level3 = parts[2]
                folder_info.append(
                    {
                        "level1": level1,
                        "level2": level2,
                        "level3": level3
                     }
                    )
            elif len(parts) == 2:
                level1 = parts[0]
                level2 = parts[1]
                folder_info.append({"level1": level1, "level2": level2})

    return folder_info

# Step 1: Read all the directories {Level1: {id of classified}, level2: 'roomName'}
folder_infos = generate_folder_info("image_set")

#Step 
for folder_info in folder_infos:

    level1 = folder_info['level1']
    level2 = folder_info['level2']

    level_heirarchy = f"{level1}/{level2}"
    pathToImageFolder = f"image_set/{level_heirarchy}"

    if 'level3' in folder_info:
        level3 = folder_info['level3']
        print("more than 3 levels", f"image_set/{level_heirarchy}/{level3}")
        pathToImageFolder = f"image_set/{level_heirarchy}/{level3}"

    """
    4 x 2

    pano_width = 1024 * 4 {pano_f.width}
    pano_height = 1024 * 2 {pano_f.height}

    actual_size = 4096 x 2048
    
    """

    print("path: ", pathToImageFolder)

    temp_image = np.array(Image.open(f"{pathToImageFolder}/pano_f.jpg"))

    fixed_image = image_list(
        [
            Image.open(f"{pathToImageFolder}/pano_f.jpg"), #Image.open('front.jpeg'),  # F
            Image.open(f"{pathToImageFolder}/pano_r.jpg"), #Image.open('right.jpeg'),  # R
            Image.open(f"{pathToImageFolder}/pano_b.jpg"), #Image.open('back.jpeg'),   # B
            Image.open(f"{pathToImageFolder}/pano_l.jpg"), #Image.open('left.jpeg'),   # L
            Image.open(f"{pathToImageFolder}/pano_u.jpg"), #Image.open('up.jpeg'),     # U
            Image.open(f"{pathToImageFolder}/pano_d.jpg") #Image.open('down.jpeg')    # D
        ]
    )

    fixed_width = temp_image.shape[1] * 4
    fixed_height = temp_image.shape[0] * 2

    pano_image = cube_to_equirectanlge(fixed_image, fixed_height, fixed_width, mode='nearest')

    img = Image.fromarray(pano_image.astype(np.uint8), 'RGB')

    print("sucking it bitch!")
    target = f"panorama_images/{level1}/"
    if not os.path.exists(target):
        os.makedirs(target)

    img.save(f"{target}/{level2}.jpg")
    print("sucked it bitch!")


"""

{x, y} STITCHED IMAGE -> Sphere {U,V} {Longitude & Latitude}

folder_info -> [list folders] -> [6 images]

image_set/{id}/room{something}/nameOfImage.jpg

"""

""""

6 - cube image to Pano: cubemap-to-pano

[6 cube maps] -> image_list -> one_stiched_image (NOT PANO)
one_stiched_image -> cube_to_equirectanlge -> Panorama Image


script to scrap images: scrap-to-CubeMap
(some-very-smart-function) -> [6-cube-map-image]


Proposal 1:
scrap-to-CubeMap -> [6-set-images] -> cubemap-to-pano -> Panorama Image

Proposal 2: (STATIC & LOCAL) (WE ARE MOVING FORWRAD WITH THIS 🚀)
Aleady downloaded pictures
[DIRECTORY] -> [List of folders] -> [nested list of each classifieds] -> [nested list of each room] -> [6-cube-map] -> cubemap-to-pano -> Panorama Image

Proposal 3: (Scrapping Dynamic) (BACKEND SIDE)
scrap-to-CubeMap -> [6-set-images] -> cubemap-to-pano -> Panorama Image

"""
