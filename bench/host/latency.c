/* Pointer-chase load-to-use latency over a range of working-set sizes.
 * Usage: latency <bytes> [<bytes> ...]   Prints one JSON object per size.
 * The chase visits cache lines in a seeded random cyclic order so hardware prefetchers
 * cannot predict the next address. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LINE 128 /* bytes; Apple M-series cache line (also safe for 64-byte lines) */

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t rng_state = 20260928u;
static uint64_t next_rand(void) { /* xorshift64 */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static double chase(size_t bytes) {
    size_t lines = bytes / LINE, stride = LINE / sizeof(uint64_t);
    if (lines < 2) return -1;
    uint64_t *buf = aligned_alloc(LINE, lines * LINE);
    size_t *order = malloc(lines * sizeof(size_t));
    if (!buf || !order) return -1;
    for (size_t i = 0; i < lines; i++) order[i] = i;
    for (size_t i = lines - 1; i > 0; i--) {
        size_t j = next_rand() % (i + 1), t = order[i];
        order[i] = order[j];
        order[j] = t;
    }
    for (size_t i = 0; i < lines; i++)
        buf[order[i] * stride] = order[(i + 1) % lines] * stride;
    size_t steps = 20 * 1000 * 1000, p = 0;
    for (size_t i = 0; i < lines; i++) p = buf[p]; /* warm-up: one full lap */
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_s();
        for (size_t i = 0; i < steps; i++) p = buf[p];
        double ns = (now_s() - t0) / steps * 1e9;
        if (ns < best) best = ns;
    }
    volatile size_t sink = p;
    (void)sink;
    free(buf);
    free(order);
    return best;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        size_t bytes = strtoull(argv[i], NULL, 10);
        printf("{\"working_set_bytes\": %zu, \"latency_ns\": %.3f}\n", bytes, chase(bytes));
        fflush(stdout);
    }
    return 0;
}
