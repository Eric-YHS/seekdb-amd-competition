/*
 * Copyright (c) 2026 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <stdint.h>

#include <string>

namespace oceanbase
{
namespace common
{
namespace obvsag
{

// Build in a separate executable to isolate hipVS and VSAG's hnswlib symbols.
int build_hipvs_hnsw(const float* vectors, const int64_t* ids, int64_t rows, int dim, int m,
                     int ef_construction, std::string& serialized, std::string& error,
                     std::string& diagnostics);

}  // namespace obvsag
}  // namespace common
}  // namespace oceanbase
