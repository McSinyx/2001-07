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

from scipy.io.wavfile import read

from .helpers import features, models

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('models', help='path to models')
    parser.add_argument('data', help='path to test data')
    args = parser.parse_args()
    choices = models(args.models)
    for audio in iglob(join(args.data, '*', '*.wav')):
        test = features(*read(audio))
        scores = {name: model.score(test) for name, model in choices.items()}
        print(audio, max(scores, key=scores.get))