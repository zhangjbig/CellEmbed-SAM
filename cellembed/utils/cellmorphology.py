import cv2
import numpy as np
import pandas as pd
import math
from skimage import measure
from scipy.spatial.distance import cdist


class CellMorphology:
    def __init__(self, contour_points, mask):
        """
        Initialize the CellMorphology class with the contour points of a cell.

        Args:
            contour_points (numpy.ndarray): Contour points of the cell.
            mask (numpy.ndarray): Binary mask for the specific cell instance (uint8).
        """
        self.contour_points = contour_points
        self.mask = mask

    def calculate_area(self):
        """
        Calculate the area of the cell using its contour.

        Returns:
            float: Area of the cell.
        """
        return cv2.contourArea(self.contour_points)

    def calculate_elongation(self):
        """
        Calculate the elongation (aspect ratio) of the cell.

        Returns:
            float: Elongation value, or None if the width is zero.
        """
        rect = cv2.minAreaRect(self.contour_points)
        box = cv2.boxPoints(rect)
        height = abs(box[1][1] - box[0][1])
        width = abs(box[1][0] - box[0][0])
        if width == 0:
            return None
        return height / width

    def calculate_compactness(self):
        """
        Calculate the compactness of the cell.

        Returns:
            float: Compactness value, or None if the perimeter is zero.
        """
        area = self.calculate_area()
        perimeter = cv2.arcLength(self.contour_points, True)
        if perimeter == 0:
            return None
        return (4 * math.pi * area) / (perimeter ** 2)

    def calculate_eccentricity(self):
        """
        Calculate the eccentricity of the cell.

        Returns:
            float: Eccentricity value, or None if the contour points are insufficient.
        """
        if len(self.contour_points) > 5:
            ellipse = cv2.fitEllipse(self.contour_points)
            axes = ellipse[1]
            width, height = axes[1] / 2, axes[0] / 2  # Width is semi-major axis, height is semi-minor axis
            if width != 0 and height != 0:
                return math.sqrt(1 - (height ** 2) / (width ** 2))
        return None

    def calculate_sphericity(self):
        """
        Calculate the sphericity of the cell.

        Returns:
            float: Sphericity value, or None if the contour points are insufficient.
        """
        if len(self.contour_points) >= 5:
            ellipse = cv2.fitEllipse(self.contour_points)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            inscribed_radius = np.sqrt(major_axis * minor_axis) / 2
            enclosing_circle = cv2.minEnclosingCircle(self.contour_points)
            enclosing_radius = enclosing_circle[1]
            if enclosing_radius != 0:
                return inscribed_radius / enclosing_radius
        return None

    def calculate_convexity(self):
        """
        Calculate the convexity of the cell.

        Returns:
            float: Convexity value, or None if the perimeter is zero.
        """
        perimeter = cv2.arcLength(self.contour_points, True)
        if perimeter == 0:
            return None
        cell_convex_hull = cv2.convexHull(self.contour_points)
        convex_hull_perimeter = cv2.arcLength(cell_convex_hull, True)
        return convex_hull_perimeter / perimeter

    def calculate_solidity(self):
        """
        Calculate the solidity of the cell.

        Returns:
            float: Solidity value, or None if the convex hull area is zero.
        """
        area = self.calculate_area()
        cell_convex_hull = cv2.convexHull(self.contour_points)
        convex_hull_area = cv2.contourArea(cell_convex_hull)
        if convex_hull_area == 0:
            return None
        return area / convex_hull_area

    def calculate_circularity(self):
        """
        Calculate the circularity of the cell.

        Returns:
            float: Circularity value, or None if the area or convex hull perimeter is zero.
        """
        area = self.calculate_area()
        if area == 0:
            return None
        cell_convex_hull = cv2.convexHull(self.contour_points)
        convex_hull_perimeter = cv2.arcLength(cell_convex_hull, True)
        if convex_hull_perimeter == 0:
            return None
        return (4 * math.pi * area) / (convex_hull_perimeter ** 2)

    def cell_tightness(self, full_label_image):
        """
        Calculate the cell tightness and average distance between cells using the full label image.

        Args:
            full_label_image (numpy.ndarray): Full instance segmentation image with all labels.

        Returns:
            tuple: Average distance between cells and tightness value.
        """
        if not self.contour_points.size:
            return None, None

        properties = measure.regionprops(full_label_image)
        if not properties:
            return None, None

        centroids = np.array([prop.centroid for prop in properties])
        areas = [prop.area for prop in properties]

        distances = cdist(centroids, centroids, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)

        average_distance = np.mean(nearest_distances)
        average_area = np.mean(areas) if areas else None

        if average_area is None or average_area == 0:
            tightness = None
        else:
            tightness = average_distance * average_distance / average_area

        return average_distance, tightness


class CellImageAnalyzer:
    def __init__(self, image_path=None, image_array=None):
        """
        Initialize the CellImageAnalyzer with a file path or numpy image array.

        Args:
            image_path (str): Optional path to the instance segmentation image.
            image_array (np.ndarray): Optional image array input (2D grayscale, uint8 or uint16).
        """
        if image_path:
            self.image_path = image_path
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        elif image_array is not None:
            image = image_array.copy()
        else:
            raise ValueError("Either image_path or image_array must be provided.")

        # Ensure image is 2D (grayscale) and uint8 or uint16
        if len(image.shape) != 2:
            raise ValueError("Input image must be grayscale (single channel).")
        if image.dtype not in [np.uint8, np.uint16]:
            raise ValueError("Image must be uint8 or uint16.")

        self.image = image
        self.stats = None
        self.morphology_stats = None

    def compute_contours_and_stats(self):
        """
        Compute contours and connected component statistics for the instance segmentation image.
        """
        # Get unique labels (excluding background 0)
        labels = np.unique(self.image)
        labels = labels[labels > 0]  # Exclude background

        self.stats = []
        for label in labels:
            # Create binary mask for the current label (uint8 for OpenCV)
            binary_mask = (self.image == label).astype(np.uint8) * 255

            # Compute connected components and stats
            _, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

            # Find contours for the current label
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

            # Store stats and contour for the current label
            if len(contours) > 0:
                # Take the largest contour (in case of multiple contours)
                contour = max(contours, key=cv2.contourArea)
                self.stats.append({
                    'label': label,
                    'area': stats[1, cv2.CC_STAT_AREA] if len(stats) > 1 else 0,
                    'left': stats[1, cv2.CC_STAT_LEFT] if len(stats) > 1 else 0,
                    'top': stats[1, cv2.CC_STAT_TOP] if len(stats) > 1 else 0,
                    'width': stats[1, cv2.CC_STAT_WIDTH] if len(stats) > 1 else 0,
                    'height': stats[1, cv2.CC_STAT_HEIGHT] if len(stats) > 1 else 0,
                    'contour': contour,
                    'mask': binary_mask
                })

    def calculate_morphology_stats(self):
        """
        Calculate morphological statistics for each cell in the image.
        """
        if self.stats is None:
            raise ValueError("Please compute cell contours and statistics first.")

        self.morphology_stats = []
        for stat in self.stats:
            contour_points = np.array(stat['contour'])
            binary_mask = stat['mask']

            if contour_points.size == 0:
                self.morphology_stats.append([None] * 10)
                continue

            cell = CellMorphology(contour_points, binary_mask)

            cell_area = cell.calculate_area()
            cell_elongation = cell.calculate_elongation()
            cell_compactness = cell.calculate_compactness()
            cell_eccentricity = cell.calculate_eccentricity()
            cell_sphericity = cell.calculate_sphericity()
            cell_convexity = cell.calculate_convexity()
            cell_solidity = cell.calculate_solidity()
            cell_circularity = cell.calculate_circularity()
            average_distance, tightness = cell.cell_tightness(self.image)  # Pass full label image

            self.morphology_stats.append([
                cell_area, cell_elongation, cell_compactness, cell_eccentricity,
                cell_sphericity, cell_convexity, cell_solidity, cell_circularity,
                average_distance, tightness
            ])

    def generate_dataframe(self):
        """
        Generate a DataFrame containing morphological statistics.

        Returns:
            pandas.DataFrame: DataFrame with the calculated morphology statistics.
        """
        if self.morphology_stats is None:
            raise ValueError("Please calculate the morphological statistics first.")

        df = pd.DataFrame(self.morphology_stats, columns=[
            'cellArea', 'cellElongation', 'cellCompactness', 'cellEccentricity',
            'cellSphericity', 'cellConvexity', 'cellSolidity', 'cellCircularity',
            'averageDistance', 'cellTightness'
        ])

        # Use original labels as index
        df.index = [stat['label'] for stat in self.stats]
        df.index.name = 'Label'
        return df


# Example usage
if __name__ == "__main__":
    # Simulate a 512x512 instance segmentation image with >256 labels
    gt_image = np.zeros((512, 512), dtype=np.uint16)
    for i in range(1, 300):  # Create 299 instances
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(mask, (np.random.randint(50, 462), np.random.randint(50, 462)), 20, 255, -1)
        gt_image[mask > 0] = i

    # Analyze the image
    analyzer = CellImageAnalyzer(image_array=gt_image)
    analyzer.compute_contours_and_stats()
    analyzer.calculate_morphology_stats()
    df = analyzer.generate_dataframe()
    print(df)