import os
import pickle

import numpy as np
from music21 import corpus, converter, stream, note, duration, interval
from music21.analysis.floatingKey import FloatingKeyException
from tqdm import tqdm


NUM_VOICES = 4

SUBDIVISION = 4  # quarter note subdivision
BEAT_SIZE = 4

SOP = 0
BASS = 1

OCTAVE = 12

PACKAGE_DIR = '.dataset/'

voice_ids_default = list(range(NUM_VOICES))  # soprano, alto, tenor, bass

SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'

batch_size = 16
timesteps = 16
voice_index = 0

def to_onehot(index, num_indexes):
    return np.array(index == np.arange(0, num_indexes),
                    dtype=np.float32)


def chorale_to_onehot(chorale, num_pitches):
    """
    chorale is time major
    :param chorale:
    :param num_pitches:
    :return:
    """
    return np.array(list(
        map(lambda time_slice: time_slice_to_onehot(time_slice, num_pitches),
            chorale)))


def time_slice_to_onehot(time_slice, num_pitches):
    l = []
    for voice_index, voice in enumerate(time_slice):
        l.append(to_onehot(voice, num_pitches[voice_index]))
    return np.concatenate(l)


def all_features(chorale, voice_index, time_index, timesteps, num_pitches,
                 num_voices):
    """
    chorale with time major
    :param chorale:
    :param voice_index:
    :param time_index:
    :param timesteps:
    :param num_pitches:
    :param num_voices:
    :return:
    """
    mask = np.array(voice_index == np.arange(num_voices), dtype=bool) == False
    num_pitches = np.array(num_pitches)

    left_feature = chorale_to_onehot(
        chorale[time_index - timesteps:time_index, :], num_pitches=num_pitches)

    right_feature = chorale_to_onehot(
        chorale[time_index + timesteps: time_index: -1, :],
        num_pitches=num_pitches)

    if num_voices > 1:
        central_feature = time_slice_to_onehot(chorale[time_index, mask],
                                               num_pitches[mask])
    else:
        central_feature = []
    # put timesteps=None to only have the current beat
    # beat is now considered as a metadata
    # beat = to_beat(time_index, timesteps=timesteps)
    label = to_onehot(chorale[time_index, voice_index],
                      num_indexes=num_pitches[voice_index])

    return (np.array(left_feature),
            np.array(central_feature),
            np.array(right_feature),
            np.array(label)
            )


def all_metadatas(chorale_metadatas, time_index=None, timesteps=None,
                  metadatas=None):
    left = []
    right = []
    center = []
    for metadata_index, metadata in enumerate(metadatas):
        left.append(list(map(
            lambda value: to_onehot(value, num_indexes=metadata.num_values),
            chorale_metadatas[metadata_index][
            time_index - timesteps:time_index])))
        right.append(list(map(
            lambda value: to_onehot(value, num_indexes=metadata.num_values),
            chorale_metadatas[metadata_index][
            time_index + timesteps: time_index: -1])))
        center.append(to_onehot(chorale_metadatas[metadata_index][time_index],
                                num_indexes=metadata.num_values))
    left = np.concatenate(left, axis=1)
    right = np.concatenate(right, axis=1)
    center = np.concatenate(center)
    return left, center, right

    

def data_iterator(batch_size, timesteps, voice_index, phase='train', percentage_train=0.8):

    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(pickled_dataset, 'rb')) 
    voice_ids = [0, 1, 2, 3]

    num_pitches = list(map(lambda x: len(x), index2notes))
    num_voices = len(voice_ids)

    total_size = len(X)
    training_size = int(round(total_size * percentage_train))
    if phase == 'train':
        chorale_indices = np.arange(training_size)
    if phase == 'test':
        chorale_indices = np.arange(training_size, total_size)
    if phase == 'all':
        chorale_indices = np.arange(total_size)

    left_features = []
    right_features = []
    central_features = []
    left_metas = []
    right_metas = []
    central_metas = []

    labels = []
    
    for i in range(0, batch_size):
        chorale_index = np.random.choice(chorale_indices)
        extended_chorale = np.transpose(X[chorale_index])
        chorale_metas = X_metadatas[chorale_index]
        padding_dimensions = (timesteps,) + extended_chorale.shape[1:]

        start_symbols = np.array(list(
            map(lambda note2index: note2index[START_SYMBOL], note2indexes)))
        end_symbols = np.array(
            list(map(lambda note2index: note2index[END_SYMBOL], note2indexes)))

        extended_chorale = np.concatenate(
            (np.full(padding_dimensions, start_symbols),
             extended_chorale,
             np.full(padding_dimensions, end_symbols)),
            axis=0)
        extended_chorale_metas = [np.concatenate((np.zeros((timesteps,)),
                                                  chorale_meta,
                                                  np.zeros((timesteps,))),
                                                 axis=0)
                                  for chorale_meta in chorale_metas]
        chorale_length = len(extended_chorale)

        time_index = np.random.randint(timesteps, chorale_length - timesteps)

        features = all_features(chorale=extended_chorale,
                                voice_index=voice_index, time_index=time_index,
                                timesteps=timesteps, num_pitches=num_pitches,
                                num_voices=num_voices)
        left_meta, central_meta, right_meta = all_metadatas(
            chorale_metadatas=extended_chorale_metas, metadatas=metadatas,
            time_index=time_index, timesteps=timesteps)

        (left_feature, central_feature, right_feature,
         label
         ) = features

        left_features.append(left_feature)
        right_features.append(right_feature)
        central_features.append(central_feature)

        left_metas.append(left_meta)
        right_metas.append(right_meta)
        central_metas.append(central_meta)
        labels.append(label) 

    left_features = np.array(left_features)
    right_features = np.array(right_features)
    central_features = np.array(central_features)
    left_metas = np.array(left_metas)
    right_metas = np.array(right_metas)
    central_metas = np.array(central_metas)
    labels = np.array(labels)

    features = {'left_features': left_features,
                                  'central_features': central_features, 
                                  'right_features': right_features,
                                  'left_metas': left_metas,
                                  'central_metas': central_metas,
                                  'right_metas': right_metas}
    labels = labels

    return features, labels

