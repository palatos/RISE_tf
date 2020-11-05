from skimage.transform import resize
import numpy as np
import random
from skimage.filters import gaussian
from tqdm import tqdm

class RISE:
    
    """
    Generate heatmap explanations for image classifiers using the RISE methodology by Petsiuk et al. 
    (reference: https://arxiv.org/abs/1806.07421)
    Generate N binary masks of initial size s by s, which are then upsampled and applied to an image.
    Elements in the initial arrays are set to 1 with probability p1. Else, they are set to 0.
    The final heatmap is generated as a linear combination of the masks.
    The weights are obtained from the softmax probabilities predicted by the base model on the masked images
    """
    def __init__(self):
        
        self.model = None
        self.input_size = None
        self.masks = None
    
    def generate_masks(self,N, s, p1):

        """
        Generate a distribution of random binary masks.

        Args:
            N: Number of masks.
            s: Size of mask before upsampling.
            p1: Probability of setting element value to 1 in the initial mask.
            verbose: Verbose level for the model prediction step.
            batch_size: Batch size for predictions.

        Returns:
            masks: The distribution of upsampled masks.
        """

        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        masks = np.empty((N, *self.input_size))

        
        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        masks = masks.reshape(-1, *self.input_size, 1)
        return masks    
        
    def explain(
        self, 
        inp, 
        model, 
        preprocessing_fn = None, 
        masks_user = None, 
        N = 2000, 
        s = 8, p1 = 
        0.5, 
        verbose = 0, 
        batch_size = 100, 
        mode = None
    ):
        """
        Generate the explanation heatmaps for all classes.

        Args:
            model: The image classifier. Typically expects a Tensorflow 2.0/Keras model or equivalent class.
            inp: The image to be explained. Expected to be in the shape used by the model, without any color
            normalization or futher preprocessing applied. Ideally the any color preprocessing is included
            within the model class/function.
            preprocessing_fn: Not implemented yet. For now preprocessing should ideally be included within the model.
            masks_user: This function calls another function to generate a mask distribution. However a user generated
            distribution of masks can be passed with this argument.
            N: Number of masks.
            s: Size of mask before upsampling.
            p1: Probability of setting element value to 1 in the initial mask.
            verbose: Verbose level for the model prediction step.
            batch_size: Batch size for predictions.
            mode (experimental): Alternative perturbation modes instead of the simple black gradation mask. 'blur'
            is a Gaussian blur, 'noise' is colored noise and 'noise_bw' is black and white noise. If None will return
            the regular black gradation perturbations. Default is None. 

        Returns:
            sal: Explanation heatmaps for all classes. For a given class_id, the heatmap can be access 
            with sal[class_id].
            masks: The distribution of masks used for generating the set of heatmaps.
        """
        self.model = model
        self.input_size = model.input_shape[1:3]

        if masks_user == None: 
            self.masks = self.generate_masks(N, s, p1)
        else:
            self.masks = masks_user #In case the user wants to pass some custom numpy array of masks.
        
        # Make sure multiplication is being done for correct axes
        
        image = inp
        fudged_image = image.copy()

        if mode == 'blur': #Gaussian blur
            fudged_image = gaussian(fudged_image, sigma=4, multichannel=True, preserve_range = True)

        elif mode == 'noise': #Colored noise
            fudged_image = np.random.normal(255/2,255/9,size = fudged_image.shape).astype('int')

        elif mode == 'noise_bw': #Grayscale noise
            fudged_image = np.random.normal(255/2,255/5,size = (fudged_image.shape[:2])).astype('int')
            fudged_image = np.stack((fudged_image,)*3, axis=-1)
            
        else:
            fudged_image = np.zeros(image.shape) #Regular perturbation with a black gradation

        
        preds = []
        
        #Doing these matrix multiplications between the masks and the image can quickly eat up memory.
        #So we multiply the image with one batch of masks at a time and later append the predictions.

        if(verbose):
            print('Using batch size: ',batch_size, flush = True)

        for i in (tqdm(range(0, N, batch_size)) if verbose else range(0, N, batch_size)):

            masks_batch = self.masks[i:min(i+batch_size, N)]
            masked = image*masks_batch + fudged_image*(1-masks_batch)
            
            to_append = model.predict(masked)

            preds.append(to_append)

        preds = np.vstack(preds)

        sal = preds.T.dot(self.masks.reshape(N, -1)).reshape(-1, *self.input_size)
        sal = sal / N / p1

        return sal, self.masks