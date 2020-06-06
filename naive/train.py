# Create models from extracted voice features
# Copyright (C) 2020  Nguyễn Gia Phong
#
# This file is part of speakerid
#
# speakerid is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# speakerid is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with speakerid.  If not, see <https://www.gnu.org/licenses/>.

from argparse import ArgumentParser
from glob import iglob
from os import makedirs
from os.path import basename, join
from pickle import dump

from numpy import concatenate
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture

from .helpers import features


def train(individuals):
    """Return an iterator of dirname and corresponding voice features.

    Each of the directory given by the iterable individuals
    should contain audio files with .wav extension.
    """
    for i in individuals:
        model = GaussianMixture(n_components=8, max_iter=200,
                                covariance_type='diag', n_init=3)
        model.fit(concatenate(tuple(features(*read(audio))
                                    for audio in iglob(join(i, '*.wav')))))
        yield basename(i), model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('source', help='path to data')
    parser.add_argument('dest', help='path to dump models')
    args = parser.parse_args()
    makedirs(args.dest, exist_ok=True)
    for file, model in train(iglob(join(args.source, '*'))):
        with open(join(args.dest, file), 'wb') as f:
            dump(model, f)
            print('written', f.name)
