# Interactive speaker identification
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

from argparse import Action, ArgumentParser
from contextlib import suppress

from numpy import concatenate
from sounddevice import InputStream, query_devices

from .helpers import features, models

NAMES = {'BI9-046': 'Vũ Vương Quốc Anh', 'BI9-084': 'Nguyễn Trường Giang',
         'BI9-108': 'Nguyễn Minh Hoàng', 'BI9-146': 'Phạm Minh Long',
         'BI9-163': 'Đỗ Đức Mạnh', 'BI9-184': 'Nguyễn Gia Phong',
         'BI9-205': 'Hoàng Nhật Tân', 'BI9-239': 'Trần Minh Vương',
         'BI8-174': 'Nguyễn An Thiết'}


class DeviceLister(Action):
    """CLI action to print available audio devices and exit."""
    def __call__(self, parser: ArgumentParser, *args, **kwargs) -> None:
        print(f'Available audio devices:\n{query_devices()}')
        parser.exit()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('models', help='path to models')
    parser.add_argument('-l', '--list-devices', action=DeviceLister, nargs=0,
                        help='print available audio devices and exit')
    parser.add_argument('-d', '--device',
                        help='input device (numeric ID or substring)')
    args = parser.parse_args()
    choices = models(args.models)
    with suppress(ValueError, TypeError): args.device = int(args.device)

    while input('test ID: '):
        audio = []
        with InputStream(
            samplerate=44100, device=args.device, channels=1, dtype='i2',
            callback=lambda i, f, t, s: audio.append(concatenate(i))):
            input('hit return to stop recording')
            test = features(44100, concatenate(audio))
        scores = {name: model.score(test) for name, model in choices.items()}
        ID = max(scores, key=scores.get)
        print(ID, NAMES[ID])
