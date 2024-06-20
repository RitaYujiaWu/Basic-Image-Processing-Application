"""
DSC 20 Project
Name(s): Yujia Wu, Zihan Xia
PID(s):  A17372175, A14970493
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # Raise exceptions here
        if not isinstance(pixels,list) \
        or len(pixels) < 1 \
        or not all([isinstance(row, list) for row in pixels]) \
        or len(set([len(row) for row in pixels])) != 1 \
        or not all([isinstance(pixel, list) for row in pixels for pixel in row]) \
        or any([len(pixel) != 3 for row in pixels for pixel in row]):
            raise TypeError()
        if not all([type(i) == int and 0 <= i <= 255 \
        for row in pixels for pixel in row for i in pixel]):
            raise ValueError()
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[i for i in pixel] for pixel in row] for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if type(row) != int or type(col) != int:
            raise TypeError()
        if row < 0 or col < 0 \
        or row >= len(self.pixels) or col >= len(self.pixels[0]):
            raise ValueError()
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if type(row) != int or type(col) != int \
        or not isinstance(new_color, tuple) \
        or len(new_color) != 3 \
        or not all([type(i) == int for i in new_color]):
            raise TypeError()
        if row < 0 or col < 0 \
        or row >= len(self.pixels) or col >= len(self.pixels[0]) \
        or any([i > 255 for i in new_color]):
            raise ValueError()
        for i in range(len(new_color)):
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        return RGBImage([[[255 - i for i in pixel] for pixel in row] for row in image.pixels])

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        return RGBImage([[[sum([i for i in pixel]) // 3] * 3 \
        for pixel in row] for row in image.pixels])

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        return RGBImage([row[::-1] for row in image.pixels[::-1]])

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        bright = [sum(value) // 3 for row in image.pixels for value in row]
        return sum(bright) // len(bright)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        if type(intensity) != int:
            raise TypeError()
        if not -255 <= intensity <= 255:
            raise ValueError()
        return RGBImage([[[0 if i + intensity < 0 \
        else 255 if i + intensity > 255 \
        else i + intensity \
        for i in pixel] for pixel in row] for row in image.pixels])

    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        def select_position(pixel_m, r, c):
            """
            Returns the desire need_to_calculate neighbors as a matrix \
            in the given matrix.
            """
            return [row[max(0, c - 1) : min(len(row), c + 1) + 1] \
            for row in pixel_m[max(0, r - 1) : min(len(pixel_m), r + 1) + 1]]
        
        r_len = len(image.pixels)
        c_len = len(image.pixels[0])
        return RGBImage([[[sum([x[i] \
        for row in select_position(image.pixels, r, c) for x in row]) \
        // (len(select_position(image.pixels, r, c)) \
        * len(select_position(image.pixels, r, c)[0])) \
        for i in range(3)] for c in range(c_len)] for r in range(r_len)])


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0
        self.coupon = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 5
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 6
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 10
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 1
        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        if self.coupon > 0:
            self.coupon -= 1
        else:
            self.cost += 5
        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if type(amount) != int:
            raise TypeError()
        if amount <= 0:
            raise ValueError()
        self.coupon += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """
        if not isinstance(chroma_image, RGBImage) or not isinstance(background_image, RGBImage):
            raise TypeError()
        if chroma_image.size() != background_image.size():
            raise ValueError()
        chroma = chroma_image.copy()
        for row in range(chroma.num_rows):
            for column in range(chroma.num_cols):
                if chroma_image.pixels[row][column] == list(color):
                    chroma.pixels[row][column] = background_image.pixels[row][column]
        return chroma

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        if not isinstance(sticker_image, RGBImage) \
        or not isinstance(background_image, RGBImage):
            raise TypeError()
        if sticker_image.size()[0] > background_image.size()[0] \
        or sticker_image.size()[1] > background_image.size()[1]:
            raise ValueError()
        if type(x_pos) != int or type(y_pos) != int:
            raise TypeError()
        if x_pos + sticker_image.size()[0] > background_image.size()[0] \
        or y_pos + sticker_image.size()[1] > background_image.size()[1]:
            raise ValueError()
        background = background_image.copy()
        for i in range(sticker_image.num_rows):
            for j in range(sticker_image.num_cols):
                background.pixels[y_pos + i][x_pos + j] = sticker_image.pixels[i][j]
        return background

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        def select_position(pixel_m, r, c):
            """
            Returns the desire need_to_calculate neighbors as a matrix \
            in the given matrix.
            """
            new_p = [[pixel_m[a][b] * 8 if a == r and b == c else pixel_m[a][b] * -1 \
            for b in range(len(pixel_m[0]))] for a in range(len(pixel_m))]

            return [row[max(0, c - 1) : min(len(row), c + 1) + 1] \
            for row in new_p[max(0, r - 1) : min(len(new_p), r + 1) + 1]]

        average_image = [[sum(col) // 3 for col in row] for row in image.pixels]
        final_p = [[sum([sum(row) for row in select_position(average_image, rows, cols)]) \
        for cols in range(len(average_image[0]))] for rows in range(len(average_image))]

        return RGBImage([[[max(0, min(final_p[rows][cols], 255))] * 3
        for cols in range(len(final_p[0]))] for rows in range(len(final_p))])


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors
        self.data = []

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        if len(data) < self.k_neighbors:
            raise ValueError()
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
        if image1.size() != image2.size():
            raise ValueError()
        r_len = len(image1.pixels)
        c_len = len(image1.pixels[0])
        return (sum([sum([sum([(image1.pixels[row][col][i] - image2.pixels[row][col][i]) ** 2 \
        for i in range(3)]) for col in range(c_len)]) for row in range(r_len)])) ** (1 / 2)

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        nums = [candidates.count(i) for i in candidates]
        return candidates[nums.index(max(nums))]

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        if not self.data:
            raise ValueError()
        distances = [(img[1], self.distance(image, img[0])) for img in self.data]
        sorted_labels = [label for label, distance in \
        sorted(distances, key=lambda x:x[1])[:self.k_neighbors]]
        return self.vote(sorted_labels)


def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label