# Copyright 2016 Oliver Cope
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from piglet.position import Position


def test_advance():
    assert Position(1, 1).advance("") == Position(1, 1)
    assert Position(1, 1).advance("f") == Position(1, 2)
    assert Position(1, 3).advance("f\n\na\nccc") == Position(4, 4)
    assert Position(1, 2).advance("f") == Position(1, 3)
