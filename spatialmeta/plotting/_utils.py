import scanpy as sc
import numpy as np
from svgpathtools import parse_path
from shapely.geometry import LineString, Point

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def get_spatial_image(adata: sc.AnnData):
    spatial_key = list(adata.uns["spatial"].keys())[0]

    if "hires" in adata.uns["spatial"][spatial_key]["images"]:
        image = adata.uns["spatial"][spatial_key]["images"]["hires"].copy()
        s = adata.uns["spatial"][spatial_key]["scalefactors"]["tissue_hires_scalef"]
    elif "lowres" in adata.uns["spatial"][spatial_key]["images"]:
        image = adata.uns["spatial"][spatial_key]["images"]["lowres"].copy()
        s = adata.uns["spatial"][spatial_key]["scalefactors"]["tissue_lowres_scalef"]
    else:
        raise ValueError("No image found")
    if all(image.flatten() <= 1):
        image = (255 * image).astype(np.uint8)
    return image, s


def get_spatial_scalefactors_dict(adata: sc.AnnData):
    spatial_key = list(adata.uns["spatial"].keys())[0]
    return adata.uns["spatial"][spatial_key]["scalefactors"]


def rgb2hex(vals, rgbtype=1):
    """Converts RGB values in a variety of formats to Hex values.

    @param  vals     An RGB/RGBA tuple
    @param  rgbtype  Valid valus are:
                         1 - Inputs are in the range 0 to 1
                       256 - Inputs are in the range 0 to 255

    @return A hex string in the form '#RRGGBB' or '#RRGGBBAA'
    """

    if len(vals) != 3 and len(vals) != 4:
        raise Exception(
            "RGB or RGBA inputs to RGBtoHex must have three or four elements!"
        )
    if rgbtype != 1 and rgbtype != 256:
        raise Exception("rgbtype must be 1 or 256!")

    # Convert from 0-1 RGB/RGBA to 0-255 RGB/RGBA
    if rgbtype == 1:
        vals = [255 * x for x in vals]

    # Ensure values are rounded integers, convert to hex, and concatenate
    return "#" + "".join(["{:02X}".format(int(round(x))) for x in vals])


def points_within_distance_along_path(path_str, spatial_coords, distance):
    # Parse SVG path string
    path = parse_path(path_str)

    # Convert SVG path to Shapely LineString
    line = LineString([(segment.start.real, segment.start.imag) for segment in path])

    # List to store points within distance along with their indices and distances
    points_within_dist = []

    # Iterate through spatial coordinates
    for idx, point in enumerate(spatial_coords):
        # Convert each point to a Shapely Point object
        point = Point(point)

        # Project the point onto the line to find the closest point on the path
        projected_point = line.interpolate(line.project(point))

        # Calculate the distance between the original point and the projected point
        dist = point.distance(projected_point)

        # If the distance is within the specified distance, add it to the list
        if dist <= distance:
            normalized_location = line.project(projected_point) / line.length * 100
            points_within_dist.append((point, idx, dist, normalized_location))

    # Sort points by their order along the path
    sorted_points_ = sorted(points_within_dist, key=lambda x: line.project(Point(x[0])))

    # Extract sorted points without distance
    sorted_points = [point for point, _, _, _ in sorted_points_]

    # Extract indices and distances
    indices = [idx for _, idx, _, _ in sorted_points_]
    distances = [dist for _, _, dist, _ in sorted_points_]
    locations = [loc for _, _, _, loc in sorted_points_]

    return sorted_points, indices, distances, locations

def points_within_distance_outside_path(path_str, spatial_coords, distance):
    # Parse SVG path string
    path = parse_path(path_str)

    # Convert SVG path to Shapely LineString
    line = LineString([(segment.start.real, segment.start.imag) for segment in path])

    # Convert SVG path to Shapely Polygon to represent the closed path
    polygon = LineString(line.coords[:])  # Closed path represented as a Polygon

    # List to store points outside the closed path along with their indices and distances
    points_outside_dist = []

    # Iterate through spatial coordinates
    for idx, point in enumerate(spatial_coords):
        # Convert each point to a Shapely Point object
        point = Point(point)

        # Project the point onto the line to find the closest point on the path
        projected_point = line.interpolate(line.project(point))

        # Calculate the distance between the original point and the projected point
        dist = point.distance(projected_point)

        # Check if the point lies outside the closed path and is greater than the specified distance
        if not point.within(polygon) and dist > distance:
            normalized_location = line.project(projected_point) / line.length * 100
            points_outside_dist.append((point, idx, dist, normalized_location))

    # Sort points by their order along the path
    sorted_points_ = sorted(points_outside_dist, key=lambda x: line.project(Point(x[0])))

    # Extract sorted points without distance
    sorted_points = [point for point, _, _, _ in sorted_points_]

    # Extract indices and distances
    indices = [idx for _, idx, _, _ in sorted_points_]
    distances = [dist for _, _, dist, _ in sorted_points_]
    locations = [loc for _, _, _, loc in sorted_points_]

    return sorted_points, indices, distances, locations
