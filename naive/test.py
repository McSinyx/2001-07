# Identify speaker using given MFCC features
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
from os.path import join
from statistics import mean

from scipy.io.wavfile import read

from .helpers import features, models


def score(choices, audios, verbose):
    """Return an iterator of the correctness of each guess."""
    for audio in audios:
        test = features(*read(audio))
        scores = {name: model.score(test) for name, model in choices.items()}
        guess = max(scores, key=scores.get)
        if verbose: print(audio, guess)
        yield guess in audio


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('models', help='path to models')
    parser.add_argument('data', help='path to test data')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='give more output')
    args = parser.parse_args()
    choices, audios = models(args.models), iglob(join(args.data, '*', '*.wav'))
    print(f'Accuracy: {mean(score(choices, audios, args.verbose)):%}')
