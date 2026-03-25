/*
  Min-cost perfect matching (assignment problem).
  Shortest-augmenting-path Hungarian algorithm (Dijkstra variant).
  No bitwise ops, multiplication, or division in source.

  Input:  "n  c11 c12 ... c1n  c21 ... cnn"  (costs 0..255)
  Output: total_cost, then n lines "row col" (1-indexed)
*/

#define MAX_N 20
#define INF 30000

static unsigned char cost[MAX_N * MAX_N];
static int u[MAX_N + 1], v[MAX_N + 1], ds[MAX_N + 1];
static int mr[MAX_N + 1], cb[MAX_N + 1];
static unsigned char way[MAX_N + 1], vis[MAX_N + 1];

__attribute__((noinline, optnone))
static void init_cb(int n) {
    int rb = 0;
    int i;
    for (i = 0; i <= MAX_N; i = i + 1) {
        cb[i] = rb;
        rb = rb + n;
    }
}

void compute(const char *input) {
    const char *p = input;
    int i, j, j0, j1, i0;

    while (*p > 0 && *p < 48) p = p + 1;
    int n = parse_int(p);
    while (*p >= 48 && *p <= 57) p = p + 1;

    init_cb(n);

    int off = 0;
    for (i = 0; i < n; i = i + 1) {
        for (j = 0; j < n; j = j + 1) {
            while (*p > 0 && *p < 48) p = p + 1;
            cost[off] = parse_int(p);
            while (*p >= 48 && *p <= 57) p = p + 1;
            off = off + 1;
        }
    }

    for (j = 0; j <= n; j = j + 1) {
        u[j] = 0;
        v[j] = 0;
        mr[j] = 0;
    }

    printf("solving %dx%d assignment problem using the Hungarian algorithm\n", n, n);

    for (i = 1; i <= n; i = i + 1) {
        mr[0] = i;

        for (j = 0; j <= n; j = j + 1) {
            ds[j] = INF;
            vis[j] = 0;
        }
        ds[0] = 0;
        j0 = 0;

        printf("\nassigning row %d\n", i);
        printf("  running Dijkstra on reduced costs...\n");

        int steps = 0;
        do {
            vis[j0] = 1;
            i0 = mr[j0];

            int ci = cb[i0 - 1];
            int ui0 = u[i0];
            int dj0 = ds[j0];
            int delta = INF;
            j1 = 0;

            for (j = 1; j <= n; j = j + 1) {
                if (vis[j] == 0) {
                    int nd = dj0 + cost[ci] - ui0 - v[j];
                    if (nd < ds[j]) {
                        ds[j] = nd;
                        way[j] = j0;
                    }
                    if (ds[j] < delta) {
                        delta = ds[j];
                        j1 = j;
                    }
                }
                ci = ci + 1;
            }

            j0 = j1;
            steps = steps + 1;
        } while (mr[j0] != 0);

        int fd = ds[j0];
        int old_cost = 0 - v[0];

        if (steps == 1) {
            printf("  explored %d column, found free col %d\n", steps, j0);
        } else {
            printf("  explored %d columns, found augmenting path to free col %d\n", steps, j0);
        }

        printf("  updating dual variables...\n");
        for (j = 0; j <= n; j = j + 1) {
            if (vis[j]) {
                int m = mr[j];
                int dj = ds[j];
                u[m] = u[m] + fd - dj;
                v[j] = v[j] - fd + dj;
            }
        }

        printf("  augmenting along path:\n");
        while (j0 != 0) {
            int prev = way[j0];
            mr[j0] = mr[prev];
            if (prev == 0) {
                printf("    assign row %d -> col %d\n", mr[j0], j0);
            } else {
                printf("    move row %d from col %d to col %d\n", mr[j0], prev, j0);
            }
            j0 = prev;
        }

        int new_cost = 0 - v[0];
        printf("  cost: %d + %d = %d (new row adds %d to total)\n",
               old_cost, new_cost - old_cost, new_cost, new_cost - old_cost);
    }

    printf("\noptimal cost: %d\n", 0 - v[0]);
    printf("final assignment:\n");
    for (j = 1; j <= n; j = j + 1) {
        printf("  row %d -> col %d (cost %d)\n", mr[j], j, cost[cb[mr[j] - 1] + j - 1]);
    }
}
