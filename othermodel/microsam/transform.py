from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import skimage.measure
import skimage.segmentation
import vigra


try:
    from affogato.affinities import compute_affinities
except ImportError:
    compute_affinities = None


class PerObjectDistanceTransform:
    """Transformation to compute normalized distances per object in a segmentation.

    Args:
        distances: Whether to compute the undirected distances.
        boundary_distances: Whether to compute the distances to the object boundaries.
        directed_distances: Whether to compute the directed distances (vector distances).
        foreground: Whether to return a foreground channel.
        apply_label: Whether to apply connected components to the labels before computing distances.
        correct_centers: Whether to correct centers that are not in the objects.
        min_size: Minimal size of objects for distance calculdation.
        distance_fill_value: Fill value for the distances outside of objects.
    """
    eps = 1e-7

    def __init__(
        self,
        distances: bool = True,
        boundary_distances: bool = True,
        directed_distances: bool = False,
        foreground: bool = True,
        instances: bool = False,
        apply_label: bool = True,
        correct_centers: bool = True,
        min_size: int = 0,
        distance_fill_value: float = 1.0,
    ):
        if sum([distances, directed_distances, boundary_distances]) == 0:
            raise ValueError("At least one of distances or directed distances has to be passed.")
        self.distances = distances
        self.boundary_distances = boundary_distances
        self.directed_distances = directed_distances
        self.foreground = foreground
        self.instances = instances

        self.apply_label = apply_label
        self.correct_centers = correct_centers
        self.min_size = min_size
        self.distance_fill_value = distance_fill_value

    def compute_normalized_object_distances(self, mask, boundaries, bb, center, distances):
        """@private
        """
        # Crop the mask and generate array with the correct center.(torch222) zhanglab@zhanglab-Super-Server:/media/zhanglab/My Passport/Mediar/instanseg-sam-main$ conda install -c ukoethe vigra=1.11.1
        # Collecting package metadata (current_repodata.json): done
        # Solving environment: -
        # The environment is inconsistent, please check the package plan carefully
        # The following packages are causing the inconsistency:
        #
        #   - defaults/linux-64::mkl_fft==1.3.11=py39h5eee18b_0
        #   - pytorch/linux-64::torchvision==0.17.2=py39_cu118
        #   - defaults/linux-64::mkl_random==1.2.8=py39h1128e8f_0
        #   - defaults/linux-64::numpy==2.0.1=py39h5f9d8c6_1
        #   - pytorch/linux-64::torchaudio==2.2.2=py39_cu118                                                                                                                                                                    failed with initial frozen solve. Retrying with flexible solve.
        # Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
        # Collecting package metadata (repodata.json): done
        # Solving environment: |
        # The environment is inconsistent, please check the package plan carefully
        # The following packages are causing the inconsistency:
        #
        #   - defaults/linux-64::mkl_fft==1.3.11=py39h5eee18b_0
        #   - pytorch/linux-64::torchvision==0.17.2=py39_cu118
        #   - defaults/linux-64::mkl_random==1.2.8=py39h1128e8f_0
        #   - defaults/linux-64::numpy==2.0.1=py39h5f9d8c6_1
        #   - pytorch/linux-64::torchaudio==2.2.2=py39_cu118                                                                                                                                                                    failed with initial frozen solve. Retrying with flexible solve.
        # Solving environment: /
        # Found conflicts! Looking for incompatible packages.                                                                                                                                                                   failed
        #
        # UnsatisfiableError: The following specifications were found
        # to be incompatible with the existing python installation in your environment:
        #
        # Specifications:
        #
        #   - vigra=1.11.1 -> python[version='2.7.*|3.5.*|3.4.*']
        #   - vigra=1.11.1 -> python[version='>=2.7,<2.8.0a0|>=3.5,<3.6.0a0|>=3.6,<3.7.0a0|>=3.8,<3.9.0a0|>=3.7,<3.8.0a0']
        #
        # Your python: python=3.9
        #
        # If python is on the left-most side of the chain, that's the version you've asked for.
        # When python appears to the right, that indicates that the thing on the left is somehow
        # not available for the python version you are constrained to. Note that conda will not
        # change your python version to a different minor version unless you explicitly specify
        # that.
        #
        # The following specifications were found to be incompatible with your system:
        #
        #   - feature:/linux-64::__glibc==2.31=0
        #   - feature:|@/linux-64::__glibc==2.31=0
        #
        # Your installed version is: 2.31
        cropped_mask = mask[bb]
        cropped_center = tuple(ce - b.start for ce, b in zip(center, bb))

        # The centroid might not be inside of the object.
        # In this case we correct the center by taking the maximum of the distance to the boundary.
        # Note: the centroid is still the best estimate for the center, as long as it's in the object.
        correct_center = not cropped_mask[cropped_center]

        # Compute the boundary distances if necessary.
        # (Either if we need to correct the center, or compute the boundary distances anyways.)
        if correct_center or self.boundary_distances:
            # Crop the boundary mask and compute the boundary distances.
            cropped_boundary_mask = boundaries[bb]
            boundary_distances = vigra.filters.distanceTransform(cropped_boundary_mask)
            boundary_distances[~cropped_mask] = 0
            max_dist_point = np.unravel_index(np.argmax(boundary_distances), boundary_distances.shape)

        # Set the crop center to the max dist point
        if correct_center:
            # Find the center (= maximal distance from the boundaries).
            cropped_center = max_dist_point

        cropped_center_mask = np.zeros_like(cropped_mask, dtype="uint32")
        cropped_center_mask[cropped_center] = 1

        # Compute the directed distances,
        if self.distances or self.directed_distances:
            this_distances = vigra.filters.vectorDistanceTransform(cropped_center_mask)
        else:
            this_distances = None

        # Keep only the specified distances:
        if self.distances and self.directed_distances:  # all distances
            # Compute the undirected ditacnes from directed distances and concatenate,
            undir = np.linalg.norm(this_distances, axis=-1, keepdims=True)
            this_distances = np.concatenate([undir, this_distances], axis=-1)

        elif self.distances:  # only undirected distances
            # Compute the undirected distances from directed distances and keep only them.
            this_distances = np.linalg.norm(this_distances, axis=-1, keepdims=True)

        elif self.directed_distances:  # only directed distances
            pass  # We don't have to do anything becasue the directed distances are already computed.

        # Add an extra channel for the boundary distances if specified.
        if self.boundary_distances:
            boundary_distances = (boundary_distances[max_dist_point] - boundary_distances)[..., None]
            if this_distances is None:
                this_distances = boundary_distances
            else:
                this_distances = np.concatenate([this_distances, boundary_distances], axis=-1)

        # Set distances outside of the mask to zero.
        this_distances[~cropped_mask] = 0

        # Normalize the distances.
        spatial_axes = tuple(range(mask.ndim))
        this_distances /= (np.abs(this_distances).max(axis=spatial_axes, keepdims=True) + self.eps)

        # Set the distance values in the global result.
        distances[bb][cropped_mask] = this_distances[cropped_mask]

        return distances

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Compute the per object distance transform.

        Args:
            labels: The segmentation

        Returns:
            The distances.
        """
        # Apply label (connected components) if specified.
        if self.apply_label:
            labels = skimage.measure.label(labels).astype("uint32")
        else:  # Otherwise just relabel the segmentation.
            labels = vigra.analysis.relabelConsecutive(labels)[0].astype("uint32")

        # Filter out small objects if min_size is specified.
        if self.min_size > 0:
            ids, sizes = np.unique(labels, return_counts=True)
            discard_ids = ids[sizes < self.min_size]
            labels[np.isin(labels, discard_ids)] = 0
            labels = vigra.analysis.relabelConsecutive(labels)[0].astype("uint32")

        # Compute the boundaries. They will be used to determine the most central point,
        # and if 'self.boundary_distances is True' to add the boundary distances.
        boundaries = skimage.segmentation.find_boundaries(labels, mode="inner").astype("uint32")

        # Compute region properties to derive bounding boxes and centers.
        ndim = labels.ndim
        props = skimage.measure.regionprops(labels)
        bounding_boxes = {
            prop.label: tuple(slice(prop.bbox[i], prop.bbox[i + ndim]) for i in range(ndim)) for prop in props
        }

        # Compute the object centers from centroids.
        centers = {prop.label: np.round(prop.centroid).astype("int") for prop in props}

        # Compute how many distance channels we have.
        n_channels = 0
        if self.distances:  # We need one channel for the overall distances.
            n_channels += 1
        if self.boundary_distances:  # We need one channel for the boundary distances.
            n_channels += 1
        if self.directed_distances:  # And ndim channels for directed distances.
            n_channels += ndim

        # Compute the per object distances.
        distances = np.full(labels.shape + (n_channels,), self.distance_fill_value, dtype="float32")
        for prop in props:
            label_id = prop.label
            mask = labels == label_id
            distances = self.compute_normalized_object_distances(
                mask, boundaries, bounding_boxes[label_id], centers[label_id], distances
            )

        # Bring the distance channel to the first dimension.
        to_channel_first = (ndim,) + tuple(range(ndim))
        distances = distances.transpose(to_channel_first)

        # Add the foreground mask as first channel if specified.
        if self.foreground:
            binary_labels = (labels > 0).astype("float32")
            distances = np.concatenate([binary_labels[None], distances], axis=0)

        if self.instances:
            distances = np.concatenate([labels[None], distances], axis=0)

        return distances
