# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.base import BaseEstimator as SklearnBaseEstimator

from el_logging import logger


class BaseEstimator(SklearnBaseEstimator):
    # http://msmbuilder.org/development/apipatterns.html

    def summarize(self):
        return 'NotImplemented'


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class Decoder(nn.Module):
    """Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class VRAE(BaseEstimator, nn.Module):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries
    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param model_dir: Download directory where models are to be dumped
    """
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',
                 n_epochs=5, dropout_rate=0., optimizer='Adam', loss='MSELoss', lr_scheduler=None,
                 cuda=False, gpu_id=0, print_every=1, save_each_epoch=10, clip=True, max_grad_norm=5, model_name='VRAE.pth', 
                 model_dir='.', train_history=None, learn_obj=None):

        super(VRAE, self).__init__()


        self.dtype = torch.FloatTensor
        self.use_cuda = cuda
        self.gpu_id = gpu_id

        if self.use_cuda and (not torch.cuda.is_available()):
            self.use_cuda = False

        if self.use_cuda:
            if (self.gpu_id < 0) or (torch.cuda.device_count() <= self.gpu_id):
                logger.warning(f"Not found GPU ID: {self.gpu_id}, changing GPU ID to 0!")
                self.gpu_id = 0

            self.device = torch.device(f"cuda:{self.gpu_id}")
            self.dtype = torch.cuda.FloatTensor
        else:
            self.device = torch.device('cpu')

        if self.use_cuda:
            with torch.cuda.device(self.device):
                self.encoder = Encoder(number_of_features = number_of_features,
                                    hidden_size=hidden_size,
                                    hidden_layer_depth=hidden_layer_depth,
                                    latent_length=latent_length,
                                    dropout=dropout_rate,
                                    block=block)

                self.lmbd = Lambda(hidden_size=hidden_size,
                                latent_length=latent_length)

                self.decoder = Decoder(sequence_length=sequence_length,
                                    batch_size = batch_size,
                                    hidden_size=hidden_size,
                                    hidden_layer_depth=hidden_layer_depth,
                                    latent_length=latent_length,
                                    output_size=number_of_features,
                                    block=block,
                                    dtype=self.dtype)
        else:
            self.encoder = Encoder(number_of_features = number_of_features,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                block=block)

            self.lmbd = Lambda(hidden_size=hidden_size,
                            latent_length=latent_length)

            self.decoder = Decoder(sequence_length=sequence_length,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                block=block,
                                dtype=self.dtype)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.save_each_epoch = save_each_epoch
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.model_dir = model_dir
        self.model_name = model_name
        self.is_fitted = False


        # callback func. for train history
        self.train_history = train_history

        self.learn_obj = learn_obj

        self.to(device=self.device)

        # optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        # set learning rate schedule
        if lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, 
                                                                  T_max=50, 
                                                                  eta_min=0.000001)
        elif lr_scheduler == 'step':
            step_size=int(n_epochs/10)
            gamma = 0.5
            if step_size <= 0:
                step_size = 1
                gamma = 1
            self.scheduler =  optim.lr_scheduler.StepLR(optimizer=self.optimizer, 
                                                        step_size=step_size, 
                                                        gamma=gamma)
        elif lr_scheduler == 'multi':
            self.scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=self.optimizer,
                                                                 lr_lambda=lambda epoch: 0.95 ** epoch)
        elif lr_scheduler == 'lambda':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                    lr_lambda=lambda epoch: 0.95 ** epoch)
        else:
            logger.info(f"LambdaLR")
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                    lr_lambda=lambda epoch: 0.95 ** epoch)

        # loss
        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(reduction="sum")
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(reduction="sum")

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder
        :param x:input tensor
        :return: the decoded output, latent vector
        """
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.decoder(latent)

        return x_decoded, latent

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the specified loss function
        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss

    def compute_loss(self, X):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration
        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)

        x_decoded, _ = self(x)
        loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach(), self.loss_fn)

        return loss, recon_loss, kl_loss, x


    def _train(self, train_loader):
        """
        For each epoch, given the batch_size, run this function batch_size * num_of_batches number of times
        :param train_loader:input train loader with shuffle
        :return:
        """
        self.train()

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        t = 0
        loss, recon_loss, kl_loss = None, None, None
        stop_learn_flag = False

        for t, X in enumerate(train_loader):

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            X = X.permute(1,0,2)

            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, _ = self.compute_loss(X)
            loss.backward()

            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)

            # accumulator
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            self.optimizer.step()

            if (t + 1) % self.print_every == 0:
                logger.info('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (t + 1, 
                                                                                          loss.item(), 
                                                                                          recon_loss.item(), 
                                                                                          kl_loss.item()))

            if self.train_history and hasattr(self.train_history, 'on_batch_end'):
                self.train_history.on_batch_end(t)

            # check stop train by force
            if self.learn_obj and self.learn_obj.stop_learn_flag is True:
                stop_learn_flag = True
                logger.debug('VRAE training stopped by force !!')
                break

        if t > 0:
            epoch_loss_average = epoch_loss / (t+1)
            epoch_recon_loss_average = epoch_recon_loss / (t+1)
            epoch_kl_loss_average = epoch_kl_loss / (t+1)
        else:
            epoch_loss_average = epoch_loss
            epoch_recon_loss_average = epoch_recon_loss
            epoch_kl_loss_average = epoch_kl_loss

        learning_rate = self.optimizer.param_groups[0]['lr']

        train_info = {
            'batch_count': t,
            'loss': epoch_loss_average,
            'recon_loss': epoch_recon_loss_average,
            'kl_loss': epoch_kl_loss_average,
            'loss_average': epoch_loss_average,
            'final_loss': None,
            'final_recon_loss': None,
            'final_kl_loss': None,
            'learning_rate': learning_rate,
            'stop_learn_flag': stop_learn_flag
        }

        if loss:
            train_info['final_loss'] = loss.item()
            train_info['final_recon_loss'] = recon_loss.item()
            train_info['final_kl_loss'] = kl_loss.item()

        return train_info

    def fit(self, dataset, save = False):
        """
        Calls `_train` function over a fixed number of epochs, specified by `n_epochs`
        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `model_dir` directory
        :return:
        """

        train_loader = DataLoader(dataset = dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)

        logger.info(f"'train_loader' length is {len(train_loader)}")

        train_info = None
        self.stop_learn_flag = False

        _epoch = 1
        _checkpoint_filename = f"{os.path.splitext(self.model_name)[0]}.checkpoint.pth"
        _checkpoint_file_path = os.path.join(self.model_dir, _checkpoint_filename)
        if os.path.isfile(_checkpoint_file_path):
            _checkpoint = torch.load(_checkpoint_file_path, map_location=self.device)
            self.load_state_dict(_checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(_checkpoint['optimizer_state_dict'])
            self.to(device=self.device)
            _epoch = _checkpoint['epoch']

        while _epoch <= self.n_epochs:

            train_info = self._train(train_loader)

            self.scheduler.step()

            if self.train_history:
                if hasattr(self.train_history, 'on_epoch_end'):
                    self.train_history.on_epoch_end(_epoch, logs=train_info)
                elif isinstance(self.train_history, dict) and ('on_epoch_end' in self.train_history) and callable(self.train_history['on_epoch_end']):
                    self.train_history['on_epoch_end'](_epoch, logs=train_info)

            if (_epoch % self.save_each_epoch) == 0:
                torch.save(
                {
                    'epoch': _epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, _checkpoint_file_path)

            if (_epoch % self.print_every) == 0:
                _train_info = train_info.copy()
                if 'batch_count' in _train_info:
                    del _train_info['batch_count']

                if 'loss' in _train_info:
                    del _train_info['loss']

                if 'stop_learn_flag' in _train_info:
                    del _train_info['stop_learn_flag']
                logger.info(f"Epoch: {_epoch}, Train info: {_train_info}")

            # check stop train by force
            if train_info and 'stop_learn_flag' in train_info:
                if train_info['stop_learn_flag'] is True:
                    logger.debug('VRAE training stopped by force !!!!!!')
                    break

            _epoch += 1

        if os.path.isfile(_checkpoint_file_path):
            os.remove(_checkpoint_file_path)

        self.is_fitted = True
        if save:
            self.save('VRAE.pth')

        if self.train_history:
            if hasattr(self.train_history, 'on_train_end'):
                self.train_history.on_train_end()

        # add final epoch
        if train_info:
            train_info['epoch'] = _epoch

        return train_info


    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function
        :param x: input batch tensor
        :return: intermediate latent vector
        """

        return self.lmbd(
                    self.encoder(
                        Variable(x.type(self.dtype), requires_grad = False)
                    )
        ).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function
        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit
        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.model_dir):
                        pass
                    else:
                        os.mkdir(self.model_dir)
                    x_decoded.dump(self.model_dir + '/latent_features.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')


    def transform(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit
        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last = False) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)

                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.model_dir):
                        pass
                    else:
                        os.mkdir(self.model_dir)
                    z_run.dump(self.model_dir + '/latent_features.pkl')
                return z_run

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above
        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    def save(self, PATH):
        """
        Pickles the model parameters to be retrieved later
        :param PATH: should contain file path
        :return: None
        """

        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: should contain file path
        :return: None
        """

        self.load_state_dict(torch.load(PATH, map_location=self.device))
        self.to(device=self.device)
        self.eval()

        self.is_fitted = True
