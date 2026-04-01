## BNN Model utilising TensorBNN.
from tensorBNN.activationFunctions import Tanh
from tensorBNN.layer import DenseLayer
from tensorBNN.network import network
from tensorBNN.likelihood import GaussianLikelihood
from tensorBNN.predictor import predictor 
from torchmetrics import MeanSquaredError as SquaredError
from tensorflow import float32
from .base import ModelBase

class BNNModel(ModelBase):
    """BNN Regression model wrapper."""
    def __init__(self, marginalization=None, seed_bnn=0, dtype=float32):
        super().__init__(marginalization)
        self.bnn = None # Placeholder, other models can be initialised before data provided. BNN cannot.
        self.seed_bnn = seed_bnn
        self.dtype = dtype
        
    def fit(self, X, y, width = 10, hidden = 2, epochs = 1000):
        """
        Inputs
        width - perceptrons per layer
        hidden - number of hidden layers
        """
        ## Variables
        inputDims = X.shape[1]
        outputDims = 1
        # width = 10 # perceptrons per layer
        # hidden = 2 # number of hidden layers
        seed = self.seed_bnn # random seed

        ## set up network object
        self.bnn = network(
            self.dtype, # network datatype
            inputDims, # dimension of input vector
            X, # training input data
            y, # training output data
            None, # validation input data
            None, # validation output data
        )
        
        ## Add layers

        # Input layer
        self.bnn.add( 
            DenseLayer( # Dense layer object
                inputDims, # Size of layer input vector
                width, # Size of layer output vector
                seed=seed, # Random seed
                dtype=self.dtype)) # Layer datatype
        self.bnn.add(
            DenseLayer.add(Tanh())) # Tanh activation function
        seed += 1000 # Increment random seed

        # hidden layers
        for n in range(hidden - 1):
            self.bnn.add(
            DenseLayer.add(DenseLayer(width,
                                    width,
                                    seed=seed,
                                    dtype=self.dtype)))
            self.bnn.add(
            DenseLayer.add(Tanh()))
            seed += 1000

        # Final layer
        self.bnn.add(
            DenseLayer.add(DenseLayer(width,
                         outputDims,
                         seed=seed,
                         dtype=dtype)))
        
        ## MCMC
        self.bnn.setupMCMC(
            0.005, # starting stepsize
            0.0025, # minimum stepsize
            0.01, # maximum stepsize
            40, # number of stepsize options in stepsize adapter
            2, # starting number of leapfrog steps
            2, # minimum number of leapfrog steps
            50, # maximum number of leapfrog steps
            1, # stepsize between leapfrog steps in leapfrog step adapter
            0.01, # hyper parameter stepsize
            5, # hyper parameter number of leapfrog steps
            20, # number of burnin epochs
            20, # number of cores
            2) # number of averaging steps for param adapters)


        likelihood = GaussianLikelihood(sd = 0.1)
        metricList = [SquaredError()]

        ## Training
        self.bnn.train(
            epochs, # epochs to train for
            2, # increment between network saves
            metricList = metricList, # List of evaluation metricx
            folderName=None) # Name of folder for saved networks

    def load_network(self, filePath):
        self.bnn = predictor(filePath,
                    dtype = self.dtype, 
                    # data type used by network
                    customLayerDict={"dense2": Dense2},
                    # A dense layer with a different 
                    # hyperprior
                    likelihood = Likelihood)
                    # The likelihood function is required to  
                    # calculate the probabilities for 
                    # re-weighting
        raise NotImplementedError("Loading pre-trained BNNs is not currently implemented.")
        #return None

    def predict_density(self, X, skip=None, context='predict'):
        dens = self.bnn.predict(X, skip, self.dtype)
        # Need a context specific way of aggregating the predictions from the network
        return dens

    def predict_mixture_params(self, X):
        # mu, std = self.gp.predict(X, return_std=True)
        # return mu, std
        raise NotImplementedError("Predicting mixture parameters is not currently implemented for BNNModel.")