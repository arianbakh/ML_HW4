import cv2
import numpy as np
import os
import sys


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
INITIAL_OUTPUT_PATH = os.path.join(BASE_DIR, 'initial_output.png')
FINAL_OUTPUT_PATH = os.path.join(BASE_DIR, 'final_output.png')


# Algorithm Parameters
NUMBER_OF_POINTS = NUMBER_OF_CLUSTERS = 10
INITIAL_LEARNING_RATE = 0.5
FINAL_LEARNING_RATE = 0.4
EPOCHS = 100


# Image Settings
IMAGE_SHAPE = (512, 512, 3)
BACKGROUND_COLOR = (255, 255, 255)
INPUT_POINT_WIDTH = 4
INPUT_POINT_COLOR = (255, 127, 127)
CLUSTER_POINT_WIDTH = 4
CLUSTER_POINT_COLOR = (127, 127, 255)  # line will also be this color


def _get_input_vectors():
    return [np.random.rand(2) for _ in range(NUMBER_OF_POINTS)]


def _init_weight_vectors(input_vector_size):
    weight_vectors = []
    for i in range(NUMBER_OF_CLUSTERS):
        weight_vectors.append(np.random.rand(input_vector_size))
    return weight_vectors


def _get_neighborhood_indices(weight_vectors, winner_index):
    neighborhood_indices = [winner_index]
    # TODO ring
    if winner_index > 0:
        neighborhood_indices.append(winner_index - 1)
    if winner_index < len(weight_vectors) - 1:
        neighborhood_indices.append(winner_index + 1)
    return neighborhood_indices


def _learn_weight_vectors(input_vectors, weight_vectors):
    alpha = INITIAL_LEARNING_RATE
    for i in range(EPOCHS):
        for input_vector in input_vectors:
            min_distance = sys.float_info.max
            winner_index = -1
            for j, weight_vector in enumerate(weight_vectors):
                distance = np.linalg.norm(input_vector - weight_vector)
                if distance < min_distance:
                    min_distance = distance
                    winner_index = j
            neighborhood_indices = _get_neighborhood_indices(weight_vectors, winner_index)
            for neighborhood_index in neighborhood_indices:
                weight_vectors[neighborhood_index] = alpha * input_vector + (1 - alpha) * weight_vectors[neighborhood_index]

        alpha = INITIAL_LEARNING_RATE - (i + 1) * (INITIAL_LEARNING_RATE - FINAL_LEARNING_RATE) / EPOCHS

    return weight_vectors


def _save_image(input_vectors, weight_vectors, image_path):
    image = np.full(IMAGE_SHAPE, BACKGROUND_COLOR, dtype=np.uint8)
    for input_vector in input_vectors:
        cv2.circle(image,
                   (
                       int(input_vector[0] * IMAGE_SHAPE[0]),
                       int(input_vector[1] * IMAGE_SHAPE[1])
                   ),
                   INPUT_POINT_WIDTH,
                   INPUT_POINT_COLOR,
                   thickness=-1)
    for weight_vector in weight_vectors:
        cv2.circle(image,
                   (
                       int(weight_vector[0] * IMAGE_SHAPE[0]),
                       int(weight_vector[1] * IMAGE_SHAPE[1])
                   ),
                   CLUSTER_POINT_WIDTH,
                   CLUSTER_POINT_COLOR,
                   thickness=-1)
    for i in range(-1, len(weight_vectors) - 1):
        cv2.line(image,
                 (
                    int(weight_vectors[i][0] * IMAGE_SHAPE[0]),
                    int(weight_vectors[i][1] * IMAGE_SHAPE[1])
                 ),
                 (
                     int(weight_vectors[i + 1][0] * IMAGE_SHAPE[0]),
                     int(weight_vectors[i + 1][1] * IMAGE_SHAPE[1])
                 ),
                 CLUSTER_POINT_COLOR)
    cv2.imwrite(image_path, image)


def run():
    input_vectors = _get_input_vectors()
    input_vector_size = len(input_vectors[0])

    weight_vectors = _init_weight_vectors(input_vector_size)
    _save_image(input_vectors, weight_vectors, INITIAL_OUTPUT_PATH)

    weight_vectors = _learn_weight_vectors(input_vectors, weight_vectors)
    _save_image(input_vectors, weight_vectors, FINAL_OUTPUT_PATH)


if __name__ == '__main__':
    run()
