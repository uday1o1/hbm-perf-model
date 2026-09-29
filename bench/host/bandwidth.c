/* Multithreaded streaming read bandwidth.
 * Usage: bandwidth <bytes> <threads> [<threads> ...]   Prints one JSON object per thread count.
 * Each thread sums a disjoint slice; the best of 7 repetitions is reported. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    const double *x;
    size_t n;
    double sum;
} slice_t;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *reader(void *arg) {
    slice_t *s = arg;
    double a = 0, b = 0, c = 0, d = 0;
    for (size_t i = 0; i + 3 < s->n; i += 4) {
        a += s->x[i];
        b += s->x[i + 1];
        c += s->x[i + 2];
        d += s->x[i + 3];
    }
    s->sum = a + b + c + d;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) return 1;
    size_t bytes = strtoull(argv[1], NULL, 10), n = bytes / sizeof(double);
    double *x = malloc(n * sizeof(double));
    if (!x) return 1;
    for (size_t i = 0; i < n; i++) x[i] = (double)i;
    for (int a = 2; a < argc; a++) {
        int threads = atoi(argv[a]);
        if (threads < 1 || threads > 64) continue;
        pthread_t th[64];
        slice_t sl[64];
        double best = 1e9, total = 0;
        for (int rep = 0; rep < 7; rep++) {
            double t0 = now_s();
            for (int t = 0; t < threads; t++) {
                sl[t].x = x + n / threads * t;
                sl[t].n = n / threads;
                pthread_create(&th[t], NULL, reader, &sl[t]);
            }
            for (int t = 0; t < threads; t++) {
                pthread_join(th[t], NULL);
                total += sl[t].sum;
            }
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("{\"threads\": %d, \"bytes\": %zu, \"read_gb_per_s\": %.3f, \"checksum\": %.1f}\n",
               threads, bytes, (double)(n / threads * threads) * sizeof(double) / best / 1e9,
               total);
        fflush(stdout);
    }
    free(x);
    return 0;
}
