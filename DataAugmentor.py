'''
MeDIT.DataAugmentor
Functions for augment the 2D or 3D numpy.array.

author: Yang Song
All right reserved
'''

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.interpolate import RegularGridInterpolator
from collections import OrderedDict

random_2d_augment = {
    'stretch_x': 0.1,
    'stretch_y': 0.1,
    'shear': 0.1,
    'shift_x': 4,
    'shift_y': 4,
    'rotate_z_angle': 20,
    'horizontal_flip': True,

    'bias_center': 0.5, # 0代表在center在中央，1代表全图像随机取，0.5代表在0.25-0.75范围内随机取
    'bias_drop_ratio': 0.5 # 随机生成0.5及以下的drop ratio
}

class AugmentParametersGenerator():
    '''
    To generate random parameters for 2D or 3D numpy array transform.
    '''
    def __init__(self):
        self.stretch_x = 0.0
        self.stretch_y = 0.0
        self.stretch_z = 0.0
        self.shear = 0.0
        self.rotate_x_angle = 0.0
        self.rotate_z_angle = 0.0
        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0
        self.horizontal_flip = False
        self.vertical_flip = False
        self.slice_flip = False

        self.bias_center = [0.0, 0.0]
        self.bias_drop_ratio = 0.0

    def RandomParameters(self, parameter_dict):
        # Stretch
        if 'stretch_x' in parameter_dict:
            self.stretch_x = np.random.randint((1 - parameter_dict['stretch_x']) * 100,
                                                                   (1 + parameter_dict['stretch_x']) * 100, 1) / 100
        else:
            self.stretch_x = 1.0

        if 'stretch_y' in parameter_dict:
            self.stretch_y = np.random.randint((1 - parameter_dict['stretch_y']) * 100,
                                                                   (1 + parameter_dict['stretch_y']) * 100, 1) / 100
        else:
            self.stretch_y = 1.0

        if 'stretch_z' in parameter_dict:
            self.stretch_z = np.random.randint((1 - parameter_dict['stretch_z']) * 100,
                                                                   (1 + parameter_dict['stretch_z']) * 100, 1) / 100
        else:
            self.stretch_z = 1.0

        # Shear and rotate
        if 'shear' in parameter_dict:
            self.shear = np.random.randint(-parameter_dict['shear'] * 100,
                                                               parameter_dict['shear'] * 100, 1) / 100
        else:
            self.shear = 0.0

        if 'rotate_z_angle' in parameter_dict:
            self.rotate_z_angle = np.random.randint(-parameter_dict['rotate_z_angle'] * 100,
                                                                        parameter_dict['rotate_z_angle'] * 100, 1) / 100
        else:
            self.rotate_z_angle = 0.0

        if 'rotate_x_angle' in parameter_dict:
            self.rotate_x_angle = np.random.randint(-parameter_dict['rotate_x_angle'] * 100,
                                                                        parameter_dict['rotate_x_angle'] * 100, 1) / 100
        else:
            self.rotate_x_angle = 0.0

        # Shift
        if 'shift_x' in parameter_dict:
            if parameter_dict['shift_x'] < 0.9999:
                self.shift_x = parameter_dict['shift_x'] * (np.random.random((1,)) - 0.5)
            else:
                self.shift_x = \
                np.random.randint(-parameter_dict['shift_x'], parameter_dict['shift_x'], (1,))[0]
        else:
            self.shift_x = 0

        if 'shift_y' in parameter_dict:
            if parameter_dict['shift_y'] < 0.9999:
                self.shift_y = parameter_dict['shift_y'] * (np.random.random((1,)) - 0.5)
            else:
                self.shift_y = \
                np.random.randint(-parameter_dict['shift_y'], parameter_dict['shift_y'], (1,))[0]
        else:
            self.shift_y = 0

        if 'shift_z' in parameter_dict:
            if parameter_dict['shift_z'] < 0.9999:
                self.shift_z = parameter_dict['shift_z'] * (np.random.random((1,)) - 0.5)
            else:
                self.shift_z = \
                np.random.randint(-parameter_dict['shift_z'], parameter_dict['shift_z'], (1,))[0]
        else:
            self.shift_z = 0

        # Flip
        if ('horizontal_flip' in parameter_dict) and (parameter_dict['horizontal_flip']):
            self.horizontal_flip = np.random.choice([True, False])
        else:
            self.horizontal_flip= False
        if ('vertical_flip' in parameter_dict) and (parameter_dict['vertical_flip']):
            self.vertical_flip = np.random.choice([True, False])
        else:
            self.vertical_flip= False
        if ('slice_flip' in parameter_dict) and (parameter_dict['slice_flip']):
            self.slice_flip = np.random.choice([True, False])
        else:
            self.slice_flip = False

        # Add Bias
        if 'bias_center' in parameter_dict:
            center_x = 0.5 + parameter_dict['bias_center'] * np.random.randint(-50, 50) / 100
            center_y = 0.5 + parameter_dict['bias_center'] * np.random.randint(-50, 50) / 100
            self.bias_center = [center_x, center_y]

        if 'bias_drop_ratio' in parameter_dict:
            self.bias_drop_ratio = parameter_dict['bias_drop_ratio'] * np.random.randint(0, 100) / 100

    def GetRandomParametersDict(self):
        return OrderedDict(sorted(self.__dict__.items(), key=lambda t: t[0]))

class DataAugmentor3D():
    '''
    To process 3D numpy array transform. The transform contains: stretch in 3 dimensions, shear along x direction,
    rotation around z and x axis, shift along x, y, z direction, and flip along x, y, z direction.
    '''
    def __init__(self):
        self.stretch_x = 1.0
        self.stretch_y = 1.0
        self.stretch_z = 1.0
        self.shear = 0.0
        self.rotate_x_angle = 0.0
        self.rotate_z_angle = 0.0
        self.shift_x = 0
        self.shift_y = 0
        self.shift_z = 0
        self.horizontal_flip = False
        self.vertical_flip = False
        self.slice_flip = False
        
    def SetParameter(self, parameter_dict):
        if 'stretch_x' in parameter_dict: self.stretch_x = parameter_dict['stretch_x']
        if 'stretch_y' in parameter_dict: self.stretch_y = parameter_dict['stretch_y']
        if 'stretch_z' in parameter_dict: self.stretch_z = parameter_dict['stretch_z']
        if 'shear' in parameter_dict: self.shear = parameter_dict['shear']
        if 'rotate_z_angle' in parameter_dict: self.rotate_z_angle = parameter_dict['rotate_z_angle']
        if 'rotate_x_angle' in parameter_dict: self.rotate_x_angle = parameter_dict['rotate_x_angle']
        if 'shift_x' in parameter_dict: self.shift_x = parameter_dict['shift_x']
        if 'shift_y' in parameter_dict: self.shift_y = parameter_dict['shift_y']
        if 'shift_z' in parameter_dict: self.shift_z = parameter_dict['shift_z']
        if 'horizontal_flip' in parameter_dict: self.horizontal_flip = parameter_dict['horizontal_flip']
        if 'vertical_flip' in parameter_dict: self.vertical_flip = parameter_dict['vertical_flip']
        if 'slice_flip' in parameter_dict: self.slice_flip = parameter_dict['slice_flip']

    def _GetTransformMatrix3D(self):
        transform_matrix = np.zeros((3, 3))
        transform_matrix[0, 0] = self.stretch_x
        transform_matrix[1, 1] = self.stretch_y
        transform_matrix[2, 2] = self.stretch_z
        transform_matrix[1, 0] = self.shear

        rotate_x_angle = self.rotate_x_angle / 180.0 * np.pi
        rotate_z_angle = self.rotate_z_angle / 180.0 * np.pi

        rotate_x_matrix = [[1, 0, 0],
                           [0, np.cos(rotate_x_angle), -np.sin(rotate_x_angle)],
                           [0, np.sin(rotate_x_angle), np.cos(rotate_x_angle)]]
        rotate_z_matrix = [[np.cos(rotate_z_angle), -np.sin(rotate_z_angle), 0],
                           [np.sin(rotate_z_angle), np.cos(rotate_z_angle), 0],
                           [0, 0, 1]]

        return transform_matrix.dot(np.dot(rotate_x_matrix, rotate_z_matrix))

    def _Shift3DImage(self, data):
        non = lambda s: s if s < 0 else None
        mom = lambda s: max(0, s)

        shifted_data = np.zeros_like(data)
        shifted_data[mom(self.shift_x):non(self.shift_x), mom(self.shift_y):non(self.shift_y), mom(self.shift_z):non(self.shift_z)] = \
            data[mom(-self.shift_x):non(-self.shift_x), mom(-self.shift_y):non(-self.shift_y), mom(-self.shift_z):non(-self.shift_z)]
        return shifted_data

    def _Flip3DImage(self, data):
        if self.horizontal_flip: data = np.flip(data, axis=1)
        if self.vertical_flip: data = np.flip(data, axis=0)
        if self.slice_flip: data = np.flip(data, axis=2)
        return data

    def ClearParameters(self):
        self.__init__()

    def _Transform3Ddata(self, data, transform_matrix, method='nearest'):
        new_data = np.copy(np.float32(data))

        image_row, image_col, image_slice = np.shape(new_data)
        x_vec = np.array(range(image_row)) - image_row // 2
        y_vec = np.array(range(image_col)) - image_col // 2
        z_vec = np.array(range(image_slice)) - image_slice // 2

        x_raw, y_raw, z_raw = np.meshgrid(x_vec, y_vec, z_vec)
        x_raw = x_raw.flatten()
        y_raw = y_raw.flatten()
        z_raw = z_raw.flatten()
        point_raw = np.transpose(np.stack([x_raw, y_raw, z_raw], axis=1))

        points = np.transpose(transform_matrix.dot(point_raw))

        my_interpolating_function = RegularGridInterpolator((x_vec, y_vec, z_vec), new_data, method=method,
                                                            bounds_error=False, fill_value=0)
        temp = my_interpolating_function(points)
        result = np.reshape(temp, (image_row, image_col, image_slice))
        result = np.transpose(result, (1, 0, 2))  # 经过插值之后, 行列的顺序会颠倒
        return result

    def Execute(self, source_data, parameter_dict={}, interpolation='linear', is_clear=False):
        if source_data.ndim != 3:
            print('Input the data with 3 dimensions!')
            return source_data
        if parameter_dict != {}:
            self.SetParameter(parameter_dict)

        transform_matrix = self._GetTransformMatrix3D()
        target_data = self._Transform3Ddata(source_data, transform_matrix=transform_matrix, method=interpolation)

        target_data = self._Shift3DImage(target_data)
        target_data = self._Flip3DImage(target_data)

        if is_clear:
            self.ClearParameters()

        return target_data

class DataAugmentor2D():
    '''
    To process 2D numpy array transform. The transform contains: stretch in x, y dimensions, shear along x direction,
    rotation, shift along x, y direction, and flip along x, y direction.
    '''
    def __init__(self):
        self.stretch_x = 1.0
        self.stretch_y = 1.0
        self.shear = 0.0
        self.rotate_z_angle = 0.0
        self.shift_x = 0
        self.shift_y = 0
        self.horizontal_flip = False
        self.vertical_flip = False
        self.bias_center = [0.5, 0.5]   # 默认bias的中心在图像中央
        self.bias_drop_ratio = 0.0

        self.is_debug = False

    def SetParameter(self, parameter_dict):
        if 'stretch_x' in parameter_dict: self.stretch_x = parameter_dict['stretch_x']
        if 'stretch_y' in parameter_dict: self.stretch_y = parameter_dict['stretch_y']
        if 'shear' in parameter_dict: self.shear = parameter_dict['shear']
        if 'rotate_z_angle' in parameter_dict: self.rotate_z_angle = parameter_dict['rotate_z_angle']
        if 'shift_x' in parameter_dict: self.shift_x = parameter_dict['shift_x']
        if 'shift_y' in parameter_dict: self.shift_y = parameter_dict['shift_y']
        if 'horizontal_flip' in parameter_dict: self.horizontal_flip = parameter_dict['horizontal_flip']
        if 'vertical_flip' in parameter_dict: self.vertical_flip = parameter_dict['vertical_flip']

        if 'bias_center' in parameter_dict: self.bias_center = parameter_dict['bias_center']
        if 'bias_drop_ratio' in parameter_dict: self.bias_drop_ratio = parameter_dict['bias_drop_ratio']

    def _GetTransformMatrix2D(self):
        transform_matrix = np.zeros((2, 2))
        transform_matrix[0, 0] = self.stretch_x
        transform_matrix[1, 1] = self.stretch_y
        transform_matrix[1, 0] = self.shear

        rotate_z_angle = self.rotate_z_angle / 180.0 * np.pi

        rotate_z_matrix = np.squeeze(np.asarray([[np.cos(rotate_z_angle), -np.sin(rotate_z_angle)],
                           [np.sin(rotate_z_angle), np.cos(rotate_z_angle)]]))

        return transform_matrix.dot(rotate_z_matrix)

    def _Shift2DImage(self, data):
        non = lambda s: s if s < 0 else None
        mom = lambda s: max(0, s)

        shifted_data = np.zeros_like(data)
        shifted_data[mom(self.shift_x):non(self.shift_x), mom(self.shift_y):non(self.shift_y),] = \
            data[mom(-self.shift_x):non(-self.shift_x), mom(-self.shift_y):non(-self.shift_y)]
        return shifted_data

    def _Flip2DImage(self, data):
        if self.horizontal_flip: data = np.flip(data, axis=1)
        if self.vertical_flip: data = np.flip(data, axis=0)
        return data

    def ClearParameters(self):
        self.__init__()

    def _Transform2Ddata(self, data, transform_matrix, method='nearest'):
        new_data = np.copy(np.float32(data))

        image_row, image_col = np.shape(new_data)
        x_vec = np.array(range(image_row)) - image_row // 2
        y_vec = np.array(range(image_col)) - image_col // 2

        x_raw, y_raw = np.meshgrid(x_vec, y_vec)
        x_raw = x_raw.flatten()
        y_raw = y_raw.flatten()
        point_raw = np.transpose(np.stack([x_raw, y_raw], axis=1))

        points = np.transpose(transform_matrix.dot(point_raw))

        my_interpolating_function = RegularGridInterpolator((x_vec, y_vec), new_data, method=method,
                                                            bounds_error=False, fill_value=0)
        temp = my_interpolating_function(points)
        result = np.reshape(temp, (image_col, image_row))
        # result = np.reshape(temp, (image_row, image_col))
        result = np.transpose(result, (1, 0))  # 经过插值之后, 行列的顺序会颠倒
        return result

    def _BiasField(self, input_shape):
        # field = a * (x - center_x) ** 2 + b * (y - center_y) ** 2 + 1
        field = np.zeros(shape=input_shape)

        center_x = int(input_shape[0] * self.bias_center[0])
        center_y = int(input_shape[1] * self.bias_center[1])
        max_x = max(input_shape[0] - center_x, center_x)
        max_y = max(input_shape[1] - center_y, center_y)

        a = -self.bias_drop_ratio / (2 * max_x ** 2)
        b = -(self.bias_drop_ratio + a * max_x ** 2) / (max_y ** 2)

        row, column = np.meshgrid(range(field.shape[0]), range(field.shape[1]))
        field = a * (row - center_x) ** 2 + b * (column - center_y) ** 2 + 1

        return field

    def _AddBias(self, data):
        field = self._BiasField(data.shape)
        if self.is_debug:
            plt.imshow(field, cmap='gray')
            plt.show()

        min_value = data.min()
        return np.multiply(field, data - min_value) + min_value

    def Execute(self, source_data, parameter_dict={}, interpolation='linear', is_clear=False, not_roi=True):
        if np.max(source_data) < 1e-6:
            return source_data

        if source_data.ndim != 2:
            print('Input the data with 2 dimensions!')
            return source_data
        if parameter_dict != {}:
            self.SetParameter(parameter_dict)

        transform_matrix = self._GetTransformMatrix2D()
        target_data = self._Transform2Ddata(source_data, transform_matrix=transform_matrix, method=interpolation)

        target_data = self._Shift2DImage(target_data)
        target_data = self._Flip2DImage(target_data)

        if self.bias_drop_ratio > 0.0 and not_roi:
            target_data = self._AddBias(target_data)

        if is_clear:
            self.ClearParameters()

        return target_data

class RandomElasticDeformation:
    """
    generate randomised elastic deformations
    along each dim for data augmentation
    """

    def __init__(self,
                 num_controlpoints=4,
                 std_deformation_sigma=15,
                 proportion_to_augment=0.5,
                 spatial_rank=3):
        """
        This layer elastically deforms the inputs,
        for data-augmentation purposes.

        :param num_controlpoints:
        :param std_deformation_sigma:
        :param proportion_to_augment: what fraction of the images
            to do augmentation on
        :param name: name for tensorflow graph
        (may be computationally expensive).
        """

        self._bspline_transformation = None
        self.num_controlpoints = max(num_controlpoints, 2)
        self.std_deformation_sigma = max(std_deformation_sigma, 1)
        self.proportion_to_augment = proportion_to_augment
        if not sitk:
            self.proportion_to_augment = -1
        self.spatial_rank = spatial_rank

    def randomise(self, images_shape):
        if self.proportion_to_augment >= 0:
            self._randomise_bspline_transformation(images_shape)
        else:
            # currently not supported spatial rank for elastic deformation
            # should support classification in the future
            print("randomising elastic deformation FAILED")
            pass

    def _randomise_bspline_transformation(self, shape):
        # generate transformation
        if len(shape) == 5:  # for niftynet reader outputs
            squeezed_shape = [dim for dim in shape[:3] if dim > 1]
        else:
            squeezed_shape = shape[:self.spatial_rank]
        itkimg = sitk.GetImageFromArray(np.zeros(squeezed_shape))
        trans_from_domain_mesh_size = \
            [self.num_controlpoints] * itkimg.GetDimension()
        self._bspline_transformation = sitk.BSplineTransformInitializer(
            itkimg, trans_from_domain_mesh_size)

        params = self._bspline_transformation.GetParameters()
        params_numpy = np.asarray(params, dtype=float)
        params_numpy = params_numpy + np.random.randn(
            params_numpy.shape[0]) * self.std_deformation_sigma

        # remove z deformations! The resolution in z is too bad
        # params_numpy[0:int(len(params) / 3)] = 0

        params = tuple(params_numpy)
        self._bspline_transformation.SetParameters(params)

    def apply_transformation_3d(self, image, interp_order=3):
        """
        Apply randomised transformation to 2D or 3D image

        :param image: 2D or 3D array
        :param interp_order: order of interpolation
        :return: the transformed image
        """
        resampler = sitk.ResampleImageFilter()
        if interp_order > 1:
            resampler.SetInterpolator(sitk.sitkBSpline)
        elif interp_order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interp_order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            return image

        squeezed_image = np.squeeze(image)
        while squeezed_image.ndim < self.spatial_rank:
            # pad to the required number of dimensions
            squeezed_image = squeezed_image[..., None]
        sitk_image = sitk.GetImageFromArray(squeezed_image)

        resampler.SetReferenceImage(sitk_image)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(self._bspline_transformation)
        out_img_sitk = resampler.Execute(sitk_image)
        out_img = sitk.GetArrayFromImage(out_img_sitk)
        return out_img.reshape(image.shape)

def ElasticAugment(input_data_list, num_controlpoints=4, std_deformation_sigma=15, proportion_to_augment=0.5,
                 spatial_rank=0, interp_order=[3]):
    if not isinstance(input_data_list, list):
        input_data_list = [input_data_list]
    if not isinstance(interp_order, list):
        interp_order = [interp_order for index in input_data_list]
    if spatial_rank == 0:
        spatial_rank = input_data_list[0].ndim


    input_data_list = [np.expand_dims(index, axis=-1) for index in input_data_list]

    rand_pairs = RandomElasticDeformation(num_controlpoints,
                                          std_deformation_sigma,
                                          proportion_to_augment,
                                          spatial_rank)

    rand_pairs.randomise(input_data_list[0].shape)
    augment_data_list = [rand_pairs.apply_transformation_3d(data, interp) for data, interp in zip(input_data_list, interp_order)]

    return [np.squeeze(data, axis=-1) for data in augment_data_list]

def main():
    # pass
    # random_params = {'stretch_x': 0.1, 'stretch_y': 0.1, 'shear': 0.1, 'rotate_z_angle': 20, 'horizontal_flip': True}
    # param_generator = AugmentParametersGenerator()
    # aug_generator = DataAugmentor2D()

    # from MeDIT.SaveAndLoad import LoadNiiData
    # _, _, data = LoadNiiData(data_path)
    # _, _, roi = LoadNiiData(roi_path)

    # data = data[..., data.shape[2] // 2]
    # roi = roi[..., roi.shape[2] // 2]

    # from Visualization import DrawBoundaryOfBinaryMask
    # from Normalize import Normalize01

    # while True:
    #     param_generator.RandomParameters(random_params)
    #     aug_generator.SetParameter(param_generator.GetRandomParametersDict())
    #
    #     new_data = aug_generator.Execute(data, interpolation_method='linear')
    #     new_roi = aug_generator.Execute(roi, interpolation_method='nearest')
    #     DrawBoundaryOfBinaryMask(Normalize01(new_data), new_roi)


    from MeDIT.UsualUse import LoadNiiData, Imshow3DArray, Normalize01
    # Test ElasticAugment
    import matplotlib.pyplot as plt
    image, _, data = LoadNiiData(r'd:\Data\PCa-Detection\JSPH-IndenpendentTest\CA\BIAN ZHONG BEN\t2_Resize.nii')
    _, _, roi = LoadNiiData(r'd:\Data\PCa-Detection\JSPH-IndenpendentTest\CA\BIAN ZHONG BEN\prostate_roi_25D.nii.gz', dtype=int)

    # data = np.concatenate((np.zeros((data.shape[0], data.shape[1], 4)),
    #                        data,
    #                        np.zeros((data.shape[0], data.shape[1], 4))), axis=2)
    while True:
        aug_data = ElasticAugment([data, roi], num_controlpoints=4, std_deformation_sigma=3, proportion_to_augment=0.5,
                                  interp_order=[3, 1])

        # plt.imshow(np.concatenate((data, aug_data, np.abs(data - aug_data)), axis=1), cmap='gray')
        # plt.show()
        Imshow3DArray(np.concatenate((Normalize01(aug_data[0]), Normalize01(data)), axis=1),
                      roi=np.concatenate((aug_data[1].astype(int), roi), axis=1))


if __name__ == '__main__':
    main()
