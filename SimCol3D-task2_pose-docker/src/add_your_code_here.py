"""
SimCol-to-3D challenge - MICCAI 2022
Challenge link: https://www.synapse.org/#!Synapse:syn28548633/wiki/617126
Task 1: Depth prediction in simulated colonoscopy
Task 2: Camera pose estimation in simulated colonoscopy

This is a dummy example to illustrate how participants should format their prediction outputs.
Please direct questions to the discussion forum: https://www.synapse.org/#!Synapse:syn28548633/discussion/default
"""
import numpy as np
import glob
from PIL import Image
import torch
from torchvision import transforms
import os
import cv2

to_tensor = transforms.ToTensor()
def euler2mat(angle):
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat
def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat
def predict_pose(im1_path, im2_path, output_folder):
    """
    param im1_path: Path to image 1
    param im2_path: Path to image 2
    param output_folder: Path to folder where output will be saved
    predict the relative pose between the image pair and save it to a .txt file.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mean = [0.705, 0.451, 0.296]
    std = [0.202, 0.161, 0.101]
    img1 = np.array(Image.open(im1_path).convert('RGB'))
    img1 = cv2.resize(np.array(img1).astype(np.float32), (int(512), int(512)), interpolation=cv2.INTER_CUBIC)
    img1 = (img1 / 255.0 - mean) / std
    img1 = np.transpose(img1, (2, 0, 1))
    tensor_img1 = (torch.from_numpy(img1).unsqueeze(0)).to(device)
    tensor_img1 = tensor_img1.type(torch.FloatTensor)

    img2 = np.array(Image.open(im2_path).convert('RGB'))
    img2 = cv2.resize(np.array(img2).astype(np.float32), (int(512), int(512)), interpolation=cv2.INTER_CUBIC)
    img2 = (img2 / 255.0 - mean) / std
    img2 = np.transpose(img2, (2, 0, 1))
    tensor_img2 = (torch.from_numpy(img2).unsqueeze(0)).to(device)
    tensor_img2 = tensor_img2.type(torch.FloatTensor)
    pose_net = torch.load(r"Posenet_epoch24").to(device)
    # print(posenetlist[k])
    pose_net.eval()
    with torch.no_grad():
        pose = pose_net(tensor_img2, tensor_img1)
    pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
    pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])

    ### replace below with your own prediction pipeline ###
    # We generate random poses as an example
    predicted_pose = pose_mat
    # Output should be a 4x4 relative camera pose [R t; 0001], where R is a 3x3 rotation matrix, and t is a 3x1
    # translation vector between the two input images. The output should represent the position of camera 2's origin in
    # camera 1's frame. Please see the file read_poses.py provided by the challenge organizers to find out how to
    # compute the relative pose between two cameras:  https://www.synapse.org/#!Synapse:syn29430445
    # Note: R should be a valid rotation matrix.

    ### Output and save your prediction in the correct format ###
    out_file = im1_path.split('/')[-1].strip('.png') + '_to_' + im2_path.split('/')[-1].strip('.png') + '.txt'
    assert predicted_pose.shape == (4,4), \
        "Wrong size of predicted pose, expected (4,4) got {}".format(list(predicted_pose.shape))
    write_file =  open(output_folder + out_file, 'w')
    write_file.write(" ".join(map(str, predicted_pose.flatten())))
    write_file.close()
    print(out_file + ' saved')
    ### Double check that the organizers' evaluation pipeline will correctly reload your poses (uncomment below) ###
    """
    read_file = open(output_folder + out_file, 'r')
    reloaded_pose = []
    for line in read_file:
        reloaded_pose.append(list(map(float, line.split())))
    read_file.close()
    reloaded_pose = np.array(reloaded_pose).reshape(4,4)
    if np.sum(np.abs(reloaded_pose - predicted_pose)) == 0:
        print('Prediction will be correctly reloaded by organizers')
    """
