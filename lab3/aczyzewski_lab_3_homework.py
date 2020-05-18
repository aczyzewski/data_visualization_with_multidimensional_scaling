#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0]/2 + 0.25, v[1]/2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)

    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green", text="eigv"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        if text is not None:
            plt.text(v[0] / 2 + 0.25, v[1] / 2, "{}{}".format(text, i), color=color)


def plot_eigenvectors(A, **kwargs):
    """Plots all eigenvectors of the given 2x2 matrix A."""

    # DONE: Zad. 4.1. Oblicz wektory własne A. Możesz wykorzystać funkcję np.linalg.eig
    eigval, eigvec = np.linalg.eig(A)
    
    # DONE: Zad. 4.1. Upewnij się poprzez analizę wykresów, że rysowane są poprawne wektory własne (łatwo tu o pomyłkę).
    visualize_vectors(eigvec.T, **kwargs)



def EVD_decomposition(A):
    # DONE: Zad. 4.2. Uzupełnij funkcję tak by obliczała rozkład EVD zgodnie z zadaniem.

    eigval, eigvec = np.linalg.eig(A)   

    K = eigvec
    L = np.diag(eigval)
    iK = np.linalg.inv(K)

    print('---')
    print('Obliczona macierz K:')
    print(K)
    
    print('Obliczona macierz L:')
    print(L)

    print('Obliczona macierz K^-1:')
    print(iK)
    
    print('Wynik mnozenia KLK^-1:')
    print(np.matmul(K, np.matmul(L, iK)))

    print('Oryginalna macierz:')
    print(A)


def plot_attractors(A, vectors, num_vectors: int = 20, num_iterations: int = 32):

    # DONE: Zad. 4.3. Uzupełnij funkcję tak by generowała wykres z atraktorami.

    # Helpers
    def normalize(v: np.array) -> np.array: 
        """ Normalizes the vector (L2) """
        return v / np.linalg.norm(v, axis=0)

    def get_similarity_matrix(vectors_a: np.array, vectors_b: np.array) -> np.array:
        """ Returns cosine similarity matrix. 
            Arguments should be column vectors.
            (vector_a -> colums, vector_b -> rows) """
        return np.array([[np.dot(v_a, v_b.T) for v_a in vectors_a.T] for v_b in vectors_b.T])

    def remove_similar_vectors(vectors: np.array) -> np.array:
        """ Removes very similar vectors from an array """
        to_delete = []
        for idx_a in range(vectors.shape[1] - 1):
            for idx_b in range(idx_a + 1, vectors.shape[1]):
                if np.isclose(vectors[:, idx_a], vectors[:, idx_b]).all():
                    to_delete.append(idx_b)
        return np.delete(vectors, to_delete, axis=1)
    
    def generate_oposed_vectors(vectors: np.array) -> np.array:
        """ Arguments should be column vectors. """
        output = []
        for vector in vectors.T:
            output.extend([vector, -vector])
        return np.array(output).T

    # Ground truth
    _, eigvec = np.linalg.eig(A)
    eigvec = remove_similar_vectors(eigvec)
    eigvec_and_opposed = generate_oposed_vectors(eigvec)

    # Define colors
    colors = []
    colors_mapping = ['red', 'orange', 'green', 'blue', 'magenta', 'cyan']

    # Plot properties
    plt.figure()
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.margins(0.05)
    plt.grid()

    # Generate vectors
    vectors = normalize(np.array(vectors_uniform(k=num_vectors)).T)
    initial_vectors = vectors.copy()

    # AAAA... Av
    for iter_idx in range(num_iterations):
        vectors = normalize(np.matmul(A, vectors))

    # Determine the the nearest attractor
    conv_sim_matrix = get_similarity_matrix(eigvec_and_opposed, vectors)
    colors = conv_sim_matrix.argmax(axis=1)

    # Plot initial vectors as quivers
    for v, color_idx in zip(initial_vectors.T, colors):
        color = colors_mapping[color_idx % len(colors)]
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.003, color=color, 
                scale_units='xy', angles='xy', scale=1, zorder=4)

    # Plot eigenvectors
    for color_idx, v in enumerate(eigvec_and_opposed.T):
        color = colors_mapping[color_idx % len(colors)]
        plt.quiver(0.0, 0.0, v[0], v[1], headwidth=5, headlength=8, width=0.008, 
                color=color, scale_units='xy', angles='xy', scale=1, zorder=4)

    if conv_sim_matrix.max(axis=1).mean() < 0.99:
        # Plot black arrows
        for v in vectors.T:
            plt.quiver(0.0, 0.0, v[0], v[1], headwidth=5, headlength=8, width=0.008, 
                    color='black', scale_units='xy', angles='xy', scale=1, zorder=4)

    plt.show()

def show_eigen_info(A, vectors):
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=8)

    A = np.array([[2, 0],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[-1, 2],
                  [2, 1]])
    show_eigen_info(A, vectors)


    A = np.array([[3, 1],
                  [0, 2]])
    show_eigen_info(A, vectors)


    A = np.array([[2, -1],
                  [1, 4]])
    show_eigen_info(A, vectors)
