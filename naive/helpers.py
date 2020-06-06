# Helper functions
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

from glob import iglob
from os.path import basename, join
from pickle import load

from python_speech_features import mfcc
from sklearn.preprocessing import scale


def features(rate, data):
    """Extract MFCC features from the given audio."""
    return scale(mfcc(data, rate, nfft=2048, appendEnergy=False))


def models(path):
    """Load models from the given directory."""
    models = {}
    for model in iglob(join(path, '*')):
        with open(model, 'rb') as f: models[basename(model)] = load(f)
    return models
