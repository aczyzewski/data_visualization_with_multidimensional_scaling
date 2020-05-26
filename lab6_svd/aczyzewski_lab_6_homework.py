import argparse
import warnings
from typing import Tuple, Callable

import numpy as np
from PIL import Image
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")

# Aliases
Matrix = np.ndarray
SVD = Tuple[Matrix, Matrix, Matrix]
EVD = Tuple[Matrix, Matrix, Matrix]


def sklearn_svd(matrix: Matrix, n_components: int = None, **kwargs) -> SVD:
    u, sigm, vt = randomized_svd(matrix, n_components=n_components, **kwargs)
    sigm = np.diag(sigm)
    return u, sigm, vt


def custom_svd(matrix: Matrix, n_components: int = None, **kwargs) -> SVD:

    def _pad_matrix(matrix: Matrix, rows: int, cols: int) -> Matrix:
        cr, cc = matrix.shape
        matrix = np.copy(matrix)

        # Add zeros
        if cr < rows:
            padding = np.zeros((rows - cr, matrix.shape[1]))
            matrix = np.concatenate((matrix, padding), axis=0)

        # Delete rows
        if cr > rows:
            matrix = matrix[:rows, :]

        # Add columns
        if cc < cols:
            padding = np.zeros((matrix.shape[0], cols - cc))
            matrix = np.concatenate((matrix, padding), axis=1)

        # Delete rows
        if cc > cols:
            matrix = matrix[:, :cols]

        return matrix

    def _calc_inv(matrix: Matrix) -> Matrix:
        with np.errstate(divide='ignore'):
            result = 1. / matrix
            result[matrix == 0] = 0
        return result

    def _evd(a: Matrix) -> EVD:
        eigval, eigvec = np.linalg.eig(a)
        return eigvec, eigval

    A = np.copy(matrix)
    n, m = A.shape
    C = A.T @ A
    eigvec, eigval = _evd(C)

    # SIGM
    eigvec = eigvec[:, np.argsort(-eigval)]
    eigval = eigval[np.argsort(-eigval)]
    eigval = eigval[eigval != 0]
    eigval = np.sqrt(eigval)
    SIGM = np.diag(eigval)

    # V
    V = eigvec
    V_SIGM = _pad_matrix(V, V.shape[0], SIGM.shape[1])

    # U
    U = (A @ V_SIGM) @ _calc_inv(SIGM)
    U = _pad_matrix(U, n, n)

    # OUTPUT
    SIGM = _pad_matrix(SIGM, n, m)
    return U[:, :n_components], SIGM[:n_components, :n_components], \
        V.T[:n_components]


def compress_layer(matrix: Matrix, method: Callable, n_components: int
                   ) -> Matrix:
    """ Compress single layer of a photo """
    u, s, v = method(matrix, n_components)
    return u @ s @ v


def compress_image(args: argparse.Namespace) -> None:
    """ Compresses an image by applying SVD decomposition """

    def rescale(x: Matrix) -> Matrix:
        return (x - x.min()) / (x.max() - x.min())

    img = np.array(Image.open(args.file)) / 255.
    n_components = args.k if args.k is not None else img.shape[1]
    svd_method = custom_svd if args.svd_method_type == 'custom' \
        else sklearn_svd

    colormap = 'RGB'
    if len(img.shape) == 2:
        colormap = 'L'
        img = img.reshape(img.shape[0], img.shape[1], 1)

    compressed_img = []
    for ch in range(img.shape[2]):
        data = compress_layer(img[:, :, ch], svd_method, n_components)
        compressed_img.append(np.expand_dims(data, 2))

    compressed_img = np.concatenate(compressed_img, axis=2)
    compressed_img = (rescale(compressed_img) * 255).astype('uint8')

    if colormap == 'L':
        compressed_img = compressed_img[:, :, 0]

    if args.output is not None:
        Image.fromarray(compressed_img).save(args.output)


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, dest='file',
                        help='Input file path.')
    parser.add_argument('-out', '--output', default=None, dest='output',
                        help='Output file path.')
    parser.add_argument('-svd', '--svd_method_type', default='custom',
                        dest='svd_method_type', choices=['custom', 'scikit'],
                        help='Method type.')
    parser.add_argument('-k', '--k', default=None, type=int, dest='k',
                        help='Compression strength')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    compress_image(args)
