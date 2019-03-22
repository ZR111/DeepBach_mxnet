# -*- coding:utf-8 -*- 
import sys
sys.path.append('..')
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
import pickle
import numpy as np
from tqdm import tqdm
import music21
import os


SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
SUBDIVISION= 4

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



def my_generator(batch_size_per_voice, model, timesteps, sequence_length, pickled_dataset,
               temperature, num_iterations, chorale_metas):

    num_pitches = [55, 57, 57, 76]
    num_pitches = np.array(num_pitches)
    num_voices = 4
    # models = models
    X, X_metadatas, voices_ids, index2notes, note2indexes, metadatas = pickle.load(
        open(pickled_dataset, 'rb'))

   
    seq = np.zeros(shape=(2 * timesteps + sequence_length, num_voices))
    for expert_index in range(num_voices):
        # Add start and end symbol + random init
        seq[:timesteps, expert_index] = [note2indexes[expert_index][
                                             START_SYMBOL]] * timesteps
        seq[timesteps:-timesteps, expert_index] = np.random.randint(
            num_pitches[expert_index],
            size=sequence_length)

        seq[-timesteps:, expert_index] = [note2indexes[expert_index][
                                              END_SYMBOL]] * timesteps
    extended_chorale_metas = [np.concatenate((np.zeros((timesteps,)),
                                              chorale_meta,
                                              np.zeros((timesteps,))),
                                             axis=0) for chorale_meta in chorale_metas]


    min_temperature = temperature
    temperature = 1.5

    for iteration in tqdm(range(num_iterations)):
        # print(iteration)
        temperature = max(min_temperature, temperature * 0.9992)  # Recuit
        time_indexes = {}
        probas = {}
        for voice_index in range(0, num_voices):
            batch_input_features = []

            time_indexes[voice_index] = []

            left_features = []
            right_features = []
            central_features = []
            left_metas = []
            right_metas = []
            central_metas = []

            for batch_index in range(batch_size_per_voice):
                time_index = np.random.randint(timesteps,
                                           sequence_length + timesteps)
                time_indexes[voice_index].append(time_index)
                (left_feature,
                central_feature,
                right_feature,
                label) = all_features(seq, voice_index, time_index, timesteps,
                                           num_pitches, num_voices)

                left_meta, central_meta, right_meta = all_metadatas(
                                           chorale_metadatas=extended_chorale_metas,
                                           metadatas=metadatas,
                                           time_index=time_index, timesteps=timesteps)

                left_features.append(left_feature)
                right_features.append(right_feature)
                central_features.append(central_feature)

                left_metas.append(left_meta)
                right_metas.append(right_meta)
                central_metas.append(central_meta)

            left_features = np.array(left_features)
            right_features = np.array(right_features)
            central_features = np.array(central_features)
            left_metas = np.array(left_metas)
            right_metas = np.array(right_metas)
            central_metas = np.array(central_metas)

            batch_input_features = {'left_features': left_features,
                                  'central_features': central_features,
                                  'right_features': right_features,
                                  'left_metas': left_metas,
                                  'central_metas': central_metas,
                                  'right_metas': right_metas} #格式为一个字典
            #model = models[voice_index]
            #model = model.load_parameters(os.path.join(
            #     'E:\zr\deepbach\DeepBach-master-gpu-58\mx_deepbach\my_model\model' + '_' + str(
            #        voice_index) + '.params'))
            probas[voice_index] = mx.nd.softmax(model[voice_index](batch_input_features)).asnumpy()

        # updates
        for voice_index in range(0, num_voices):
                for batch_index in range(batch_size_per_voice):
                    probas_pitch = probas[voice_index][batch_index]

                    # use temperature
                    probas_pitch = np.log(probas_pitch) / temperature
                    probas_pitch = np.exp(probas_pitch) / np.sum(
                        np.exp(probas_pitch)) - 1e-7

                    # pitch can include slur_symbol
                    pitch = np.argmax(np.random.multinomial(1, probas_pitch))

                    seq[time_indexes[voice_index][
                            batch_index], voice_index] = pitch

    return seq[timesteps:-timesteps, :]


def indexed_chorale_to_score(seq, pickled_dataset):
    """

    :param seq: voice major
    :param pickled_dataset:
    :return:
    """
    _, _, _, index2notes, note2indexes, _ = pickle.load(
        open(pickled_dataset, 'rb'))
    num_pitches = list(map(len, index2notes))
    slur_indexes = list(map(lambda d: d[SLUR_SYMBOL], note2indexes))

    score = music21.stream.Score()
    for voice_index, v in enumerate(seq):
        part = music21.stream.Part(id='part' + str(voice_index))
        dur = 0
        f = music21.note.Rest()
        for k, n in enumerate(v):
            # if it is a played note
            if not n == slur_indexes[voice_index]:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur / SUBDIVISION)
                    part.append(f)

                dur = 1
                f = standard_note(index2notes[voice_index][n])
            else:
                dur += 1
        # add last note
        f.duration = music21.duration.Duration(dur / SUBDIVISION)
        part.append(f)
        score.insert(part)
    return score

def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return music21.note.Rest()
    # treat other additional symbols as rests
    if note_or_rest_string == START_SYMBOL or note_or_rest_string == END_SYMBOL:
        return music21.note.Rest()
    if note_or_rest_string == SLUR_SYMBOL:
        print('Warning: SLUR_SYMBOL used in standard_note')
        return music21.note.Rest()
    else:
        return music21.note.Note(note_or_rest_string)
