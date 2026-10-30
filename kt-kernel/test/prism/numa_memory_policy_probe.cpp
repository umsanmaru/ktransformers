// Standalone policy probe: run with KT_NUMA_MEMORY_POLICY=preferred or unset.
#include "cpu_backend/worker_pool.h"
#include <numaif.h>
#include <cassert>

int main() {
  const bool preferred = kt_preferred_numa_memory();
  for (int node = 0; node < numa_num_configured_nodes(); ++node) {
    for (int path = 0; path < 2; ++path) {
      if (path == 0) set_to_numa(node);
      else set_memory_to_numa(node);
      int mode = -1;
      unsigned long mask = 0;
      assert(get_mempolicy(&mode, &mask, sizeof(mask) * 8, nullptr, 0) == 0);
      assert(mode == (preferred ? MPOL_PREFERRED : MPOL_BIND));
      assert(mask == (1UL << node));
      printf("node=%d path=%d mode=%d mask=%lu OK\n", node, path, mode, mask);
    }
  }
}
