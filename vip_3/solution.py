import numpy as np
import ps_utils as ps
import scipy.io
import custom_utils as cs


class AssignmentModel:

    def __init__(self, model_filename):
        dataset = scipy.io.loadmat(model_filename)

        self.I = dataset['I']
        self.mask = dataset['mask']
        self.S = dataset['S']
        self.image_shape = self.I[:, :, 0].shape

        self.albedo = None
        self.M = None
        self.norms = None
        self.z_depth = None

    def compute(self, mode='inverse', threshold=1.0, smoothing=False, smooth_iters=100):
        if mode not in ['inverse', 'ransac']:
            raise ValueError('Incorrect "mode" parameter. Possible values: inverse, ransac')

        self.albedo = self._calc_albedo_inverse() if mode == 'inverse' else self._calc_albedo_ransac(threshold)
        self.z_depth = self._calc_z_depth(smoothing, smooth_iters)

    def plot(self, mode='mayavi', temp_fix_func=np.min):
        if mode not in ['mayavi', 'matplotlib']:
            raise ValueError('Incorrect "mode" parameter. Possible values: inverse, ransac')

        if self.albedo is None or self.z_depth is None:
            raise ValueError('You need to call the .compute(...) method first!')

        model_images = self.get_images()[:8]  # plot up to 8 images from the dataset, no more
        labels = ['Image #' + str(i+1) for i in range(len(model_images))]

        model_images.append(self.albedo)
        model_images += list(self.norms)

        labels += ['Albedo', 'Norm Y', 'Norm X', 'Norm Z']
        cs.plot(model_images, labels=labels, cols=4, fontsize=8, figsize=(10, 10))

        if mode == 'matplotlib':
            cs.display_surface_matplotlib_fixed(self.z_depth)
        else:
            fixed_z = np.copy(self.z_depth)
            fixed_z[self.mask == 0] = temp_fix_func(fixed_z[self.mask > 0])
            ps.display_surface(fixed_z, self.albedo)

    def get_images(self):
        return [self.I[:, :, i] for i in range(self.I.shape[2])]

    def _calc_albedo_inverse(self):
        J = self.I[self.mask > 0].T.astype(float)
        self.M = np.linalg.pinv(self.S) @ J

        albedo = np.zeros(self.image_shape)
        albedo[self.mask > 0] = np.sqrt(np.sum(self.M**2, axis=0))

        return albedo

    def _calc_albedo_ransac(self, threshold):
        J = self.I[self.mask > 0].T.astype(float)
        self.M = np.zeros((3, J.shape[1]))

        for i in range(J.shape[1]):
            self.M[:, i] = ps.ransac_3dvector((J[:, i], self.S), threshold=threshold, verbose=0)[0]

        albedo = np.zeros(self.image_shape)
        albedo[self.mask > 0] = np.linalg.norm(self.M, axis=0)

        return albedo

    def _calc_z_depth(self, smoothing, smooth_iters):
        # make sure there are no zeros in M or in the albedo
        # zeros in M will make ps.unbiased_integrate fail (division by zero inside the function)
        # zeros in the albedo cause NaN values in the normalized norms (division by zero)
        # this wouldn't matter much if mayavi wasn't buggy with NaN scalars passed into the mlab.mesh function
        # right now if there is at least 1 NaN in the whole of z depth field, mayavi won't print it
        # (mayavi version 4.7.2, Linux & Mac, but works on Windows)

        modulated_m = self.M[:]
        modulated_m[modulated_m == 0] = 1

        albedo = self.albedo[self.mask > 0]
        albedo[albedo == 0] = 1

        norms_normalized = modulated_m / albedo
        self.norms = []

        for norm in norms_normalized:
            unpacked_norm = np.ones(self.image_shape)
            unpacked_norm[self.mask > 0] = norm
            self.norms.append(unpacked_norm)

        if smoothing:
            self.norms = ps.smooth_normal_field(self.norms[0], self.norms[1], self.norms[2], self.mask, iters=smooth_iters)

        return ps.unbiased_integrate(self.norms[0], self.norms[1], self.norms[2], self.mask)


class AssignmentSection:
    @staticmethod
    def beethoven():
        print('Beethoven: computing norms using pseudo-inverse...', end='', flush=True)
        model = AssignmentModel('./data/Beethoven.mat')
        model.compute()
        model.plot(temp_fix_func=np.max)
        print('DONE', flush=True)

    @staticmethod
    def mate_vase():
        print('Mate Vase: computing norms using pseudo-inverse...', end='', flush=True)
        model = AssignmentModel('./data/mat_vase.mat')
        model.compute()
        model.plot()
        print('DONE', flush=True)

    @staticmethod
    def shiny_vase():
        print('Shiny Vase: computing norms using pseudo-inverse...', end='', flush=True)
        model = AssignmentModel('./data/shiny_vase.mat')
        model.compute()
        model.plot()
        print('DONE', flush=True)

        print('Shiny Vase: computing norms using RANSAC estimation...', end='', flush=True)
        model.compute(mode='ransac')
        model.plot()
        print('DONE', flush=True)

        # experiment with smoothing
        print('Shiny Vase: computing norms using pseudo-inverse with norm smoothing (100 iters)...', end='', flush=True)
        model.compute(smoothing=True)
        model.plot()
        print('DONE', flush=True)

        print('Shiny Vase: computing norms using RANSAC estimation with norm smoothing (100 iters)...', end='', flush=True)
        model.compute(smoothing=True, mode='ransac')
        model.plot()
        print('DONE', flush=True)

    @staticmethod
    def shiny_vase_two():
        print('Shiny Vase2: computing norms using pseudo-inverse...', end='', flush=True)
        model = AssignmentModel('./data/shiny_vase2.mat')
        model.compute()
        model.plot()
        print('DONE', flush=True)

        print('Shiny Vase2: computing norms using RANSAC estimation (threshold 1.5 <= 2)...', end='', flush=True)
        model.compute(mode='ransac', threshold=1.5)
        model.plot()
        print('DONE', flush=True)

        # experiment with smoothing
        print('Shiny Vase2: computing norms using pseudo-inverse with norm smoothing (500 iters)...', end='', flush=True)
        model.compute(smoothing=True, smooth_iters=500)
        model.plot()
        print('DONE', flush=True)

        print('Shiny Vase2: computing norms using RANSAC estimation with norm smoothing (10 iters)...', end='', flush=True)
        model.compute(smoothing=True, mode='ransac', smooth_iters=10)
        model.plot()
        print('DONE', flush=True)

    @staticmethod
    def buddha():
        print('Buddha: computing norms using pseudo-inverse...', end='', flush=True)
        model = AssignmentModel('./data/Buddha.mat')
        model.compute()
        model.plot()
        print('DONE', flush=True)

        print('Buddha: computing norms using RANSAC estimation (threshold 30 >= 25, give it time)...', end='', flush=True)
        model = AssignmentModel('./data/Buddha.mat')
        model.compute(mode='ransac', threshold=30)
        model.plot(temp_fix_func=lambda x: 0)
        print('DONE', flush=True)

        # experiment with smoothing
        print('Buddha: computing norms using pseudo-inverse with norm smoothing (20 iters)...', end='', flush=True)
        model.compute(smoothing=True, smooth_iters=20)
        model.plot()
        print('DONE', flush=True)

        print('Buddha: computing norms using RANSAC estimation (threshold 30 >= 25) with norm smoothing (150 iters)...',
              end='', flush=True)
        model.compute(mode='ransac', threshold=30, smoothing=True, smooth_iters=150)
        model.plot()
        print('DONE', flush=True)

    @staticmethod
    def face():
        print('Face: computing norms using RANSAC estimation (threshold 10) with norm smoothing (200 iters)...', end='', flush=True)
        model = AssignmentModel('./data/face.mat')
        model.compute(mode='ransac', threshold=10, smoothing=True, smooth_iters=200)
        model.plot()
        print('DONE', flush=True)

        print('Face: computing norms using RANSAC estimation (threshold 10) with norm smoothing (2 iters)...', end='', flush=True)
        model.compute(mode='ransac', threshold=10, smoothing=True, smooth_iters=2)
        model.plot()
        print('DONE', flush=True)
