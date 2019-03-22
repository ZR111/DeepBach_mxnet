# -*- coding:utf-8 -*- 
import argparse
import sys
sys.path.append('')
import mxnet as mx
from mxnet import gluon
#from gluon import data as gdata
from mxnet import ndarray as nd
from mxnet import autograd
import numpy as np
import os
import pickle
#from .model_manager.py import *
#from .data_utils.py import *
from my_dataset import data_iterator
from my_generation import my_generator
from my_generation import indexed_chorale_to_score
import music21

num_pitches = [55, 57, 57, 76]
num_voices = 4
pickled_dataset = ''


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',
                        help="model's range (default: %(default)s)",
                        type=int, default=16)
    parser.add_argument('-b', '--batch_size_train',
                        help='batch size used during training phase (default: %(default)s)',
                        type=int, default=128)
    parser.add_argument('-bg', '--batch_size_generation',
                        help='batch size per voice during generating phase(default=1)',
                        type=int, default=2)
    parser.add_argument('-t', '--train',
                        help='whether training stage',
                        type=bool, default=0)
    parser.add_argument('--validation_steps',
                        help='number of validation steps (default: %(default)s)',
                        type=int, default=20)
    parser.add_argument('-u', '--num_units_lstm', nargs='+',
                        help='number of lstm units (default: %(default)s)',
                        type=int, default=[200, 200])
    parser.add_argument('-d', '--num_dense',
                        help='size of non recurrent hidden layers (default: %(default)s)',
                        type=int, default=200)
    parser.add_argument('-n', '--name',
                        help='model base name (default: %(default)s)',
                        choices=['deepbach', 'skip'],
                        type=str, default='my_deepbach')
    parser.add_argument('-i', '--num_iterations',
                        help='number of gibbs iterations (default: %(default)s)',
                        type=int, default=20000)
    parser.add_argument('-e', '--epochs', nargs='?', default=2000,
                        help='train models for N epochs (default: 15)',
                        const=15, type=int)
    parser.add_argument('-p', '--parallel', nargs='?',
                        help='number of parallel updates (default: 16)',
                        type=int, const=16, default=1)
    parser.add_argument('--overwrite',
                        help='overwrite previously computed models',
                        action='store_true')
    parser.add_argument('-m', '--midi_file', nargs='?',
                        help='relative path to midi file',
                        type=str, const='datasets/god_save_the_queen.mid')
    parser.add_argument('-l', '--length',
                        help='length of unconstrained generation',
                        type=int, default=160)
    parser.add_argument('--ext',
                        help='extension of model name',
                        type=str, default='')
    parser.add_argument('-o', '--output_file', nargs='?',
                        help='path to output file',
                        type=str, default='',
                        const='generated_examples/example.mid')
    parser.add_argument('--dataset', nargs='?',
                        help='path to dataset folder',
                        type=str, default='')
    parser.add_argument('-r', '--reharmonization', nargs='?',
                        help='reharmonization of a melody from the corpus identified by its id',
                        type=int)
    args = parser.parse_args()
    print(args)

    train = args.train
    timesteps = args.timesteps
    batch_size = args.batch_size_train
    num_epochs = args.epochs
    sequence_length = args.length
    num_iterations = args.num_iterations
    batch_size_per_voice = args.batch_size_generation

    if train:
        for voice_index in range(0, num_voices):
            print('voice_index is ', voice_index)
            model = my_deepbach(num_pitches[voice_index])
            model.initialize()
            softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
            trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.01})
            model = train_my_deepbach(model, data_iterator, num_epochs , batch_size , timesteps = 16,
                                      voice_index = voice_index, loss = softmax_cross_entropy, trainer=trainer)

    else:
        models = []
        for voice_index in range(0, num_voices):
            model = my_deepbach(num_pitches=num_pitches[voice_index])
            model.load_parameters(os.path.join('./my_model/' + 'model_' + str(voice_index) + '(epoch = 5000).params'))
            models.append(model) 

        temperature = 1.
        metadatas = pickle.load(open('./new_dataset/metadatas.pickle', 'rb'))
        chorale_metas = [metas.generate(sequence_length) for metas in metadatas]
        seq = my_generator(batch_size_per_voice, timesteps=timesteps, model=models, sequence_length=sequence_length, pickled_dataset=pickled_dataset,
                            temperature=temperature, num_iterations=num_iterations, chorale_metas=chorale_metas)

        #np.save("seq_0.npy", seq)
        score = indexed_chorale_to_score(np.transpose(seq, axes=(1, 0)), pickled_dataset=pickled_dataset)
        score.show()
        mf = music21.midi.translate.music21ObjectToMidiFile(score)
        output_file = './results/seq.mid'
        mf.open(output_file, 'wb')
        mf.write()
        mf.close()
        print("File " + output_file + " written")


class my_deepbach(gluon.nn.Block):
    def __init__(self, num_pitches, **kwargs):
         super(my_deepbach, self).__init__(**kwargs)
         with self.name_scope():
             self.left_dense = gluon.nn.Dense(200, flatten=True)
             self.left_lstm = gluon.rnn.LSTM(200, num_layers=2, layout='TNC', dropout=0, bidirectional=False)

             self.right_dense = gluon.nn.Dense(200, flatten=True)
             self.right_lstm = gluon.rnn.LSTM(200, num_layers=2, layout='TNC', dropout=0, bidirectional=False)

             self.center_1 = gluon.nn.Dense(200, flatten=False, activation='relu')
             self.center_2 = gluon.nn.Dense(200, flatten=False, activation='relu')

             self.predictions = gluon.nn.Dense(200, activation='relu')
             self.pitch_prediction = gluon.nn.Dense(num_pitches)

    def forward(self, x): 
        left_input = np.concatenate((x['left_features'], x['left_metas']), axis=2)
        right_input = np.concatenate((x['right_features'], x['right_metas']), axis=2)
        central_input = np.concatenate((x['central_features'], x['central_metas']), axis=1)

        #embedding_left = self.left_dense(mx.nd.array(left_input.transpose(1, 0, 2)))
        embedding_left = self.left_dense(mx.nd.array(left_input))
        predictions_left = self.left_lstm(nd.expand_dims(embedding_left, axis=0))
        predictions_left = nd.flatten(predictions_left.transpose((1, 0, 2)))

        #embedding_right = self.right_dense(mx.nd.array(right_input.transpose(1, 0, 2)))
        embedding_right = self.right_dense(mx.nd.array(right_input))
        predictions_right = self.right_lstm(nd.expand_dims(embedding_right, axis=0))
        predictions_right = nd.flatten(predictions_right.transpose((1, 0, 2)))

        #prediction_center0 = self.center_1(mx.nd.array(np.expand_dims(central_input, axis=1)))
        #predictions_center = self.center_2(prediction_center0)
        #predictions_center = nd.flatten(predictions_center)

        prediction_center0 = self.center_1(mx.nd.array(central_input)) 
        predictions_center = self.center_2(prediction_center0)

        prediction = nd.concat(predictions_left, predictions_center, predictions_right)
        predictions = self.predictions(prediction)
        pitch_prediction = self.pitch_prediction(predictions)
        #pitch_prediction = mx.nd.softmax(pitch_prediction)

        return pitch_prediction


def train_my_deepbach(model, data_iter, num_epochs, batch_size, timesteps,voice_index, loss, trainer=None): 

      for epoch in range(1, num_epochs + 1):
        #for X, y in data_iter(batch_size, timesteps, voice_index):
        data_iter = data_iterator(batch_size, timesteps, voice_index)

        with autograd.record():
            X = data_iter[0]
            y = data_iter[1]
            y = nd.array(y)
            prediction = model(X)
            l = loss(prediction, y) 
            l.backward() 
        trainer.step(batch_size)
        print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
      filename = os.path.join('E:/zr/deepbach/DeepBach-master-gpu-58/mx_deepbach/my_model/model_'
                              + str(voice_index) + '(epoch = ' +str(num_epochs) + 'last_loss=%f).params'%l.mean().asnumpy())
      model.save_parameters(filename)

if __name__ == '__main__':
    main()


            










    