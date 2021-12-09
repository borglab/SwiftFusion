// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import ArgumentParser
import PenguinParallelWithFoundation

struct Scripts: ParsableCommand {
  static var configuration = CommandConfiguration(
    subcommands: [Brando01.self, Brando02.self, Brando03.self, Brando04.self,
    Brando05.self, Brando06.self, Brando07.self, Brando08.self,
    Brando10.self, Brando11.self, Brando12.self, Brando13.self, Brando14.self, Brando15.self, Brando16.self, Andrew01.self, 
    Andrew05.self, Andrew06.self, Andrew07.self, Andrew08.self, Fan01.self, Fan02.self, Fan03.self, Fan04.self, Fan05.self, 
    Fan10.self, Fan12.self, Fan13.self, Fan14.self,
                  Frank01.self, Frank02.self, Frank03.self, Frank04.self])
}

// It is important to set the global threadpool before doing anything else, so that nothing
// accidentally uses the default threadpool.
ComputeThreadPools.global =
  NonBlockingThreadPool<PosixConcurrencyPlatform>(name: "mypool", threadCount: 12)

Scripts.main()
