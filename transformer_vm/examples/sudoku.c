/*
  Constraint-propagation Sudoku solver (Norvig-style).
  Optimized for restricted WASM ISA: ADD/SUB/LOAD8/STORE8 only.

  Input:  81-char string (0 = empty)
  Output: chain-of-thought reasoning + 81-char solution
*/

static const unsigned short db[9] = {0, 81, 162, 243, 324, 405, 486, 567, 648};
static const unsigned char d27[9] = {0, 27, 54, 81, 108, 135, 162, 189, 216};

static const unsigned char unit_off[27] = {
    0, 9, 18, 27, 36, 45, 54, 63, 72,
    81, 90, 99, 108, 117, 126, 135, 144, 153,
    162, 171, 180, 189, 198, 207, 216, 225, 234
};

static const unsigned char ucells[243] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8,
     9,10,11,12,13,14,15,16,17,
    18,19,20,21,22,23,24,25,26,
    27,28,29,30,31,32,33,34,35,
    36,37,38,39,40,41,42,43,44,
    45,46,47,48,49,50,51,52,53,
    54,55,56,57,58,59,60,61,62,
    63,64,65,66,67,68,69,70,71,
    72,73,74,75,76,77,78,79,80,
     0, 9,18,27,36,45,54,63,72,
     1,10,19,28,37,46,55,64,73,
     2,11,20,29,38,47,56,65,74,
     3,12,21,30,39,48,57,66,75,
     4,13,22,31,40,49,58,67,76,
     5,14,23,32,41,50,59,68,77,
     6,15,24,33,42,51,60,69,78,
     7,16,25,34,43,52,61,70,79,
     8,17,26,35,44,53,62,71,80,
     0, 1, 2, 9,10,11,18,19,20,
     3, 4, 5,12,13,14,21,22,23,
     6, 7, 8,15,16,17,24,25,26,
    27,28,29,36,37,38,45,46,47,
    30,31,32,39,40,41,48,49,50,
    33,34,35,42,43,44,51,52,53,
    54,55,56,63,64,65,72,73,74,
    57,58,59,66,67,68,75,76,77,
    60,61,62,69,70,71,78,79,80,
};

static const unsigned char cu0[81] = {
    0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,5,
    6,6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7,7,
    8,8,8,8,8,8,8,8,8
};
static const unsigned char cu1[81] = {
     9,10,11,12,13,14,15,16,17,  9,10,11,12,13,14,15,16,17,
     9,10,11,12,13,14,15,16,17,  9,10,11,12,13,14,15,16,17,
     9,10,11,12,13,14,15,16,17,  9,10,11,12,13,14,15,16,17,
     9,10,11,12,13,14,15,16,17,  9,10,11,12,13,14,15,16,17,
     9,10,11,12,13,14,15,16,17
};
static const unsigned char cu2[81] = {
    18,18,18,19,19,19,20,20,20, 18,18,18,19,19,19,20,20,20,
    18,18,18,19,19,19,20,20,20, 21,21,21,22,22,22,23,23,23,
    21,21,21,22,22,22,23,23,23, 21,21,21,22,22,22,23,23,23,
    24,24,24,25,25,25,26,26,26, 24,24,24,25,25,25,26,26,26,
    24,24,24,25,25,25,26,26,26
};

static unsigned char cand[729];
static unsigned char cnt[81];
static unsigned char ucnt[243];

#define TRAIL_MAX 3000
static unsigned char tr_c[TRAIL_MAX];
static unsigned char tr_d[TRAIL_MAX];
static int tr_top;

#define WORK_MAX 6000
static unsigned char wk_c[WORK_MAX];
static unsigned char wk_d[WORK_MAX];
static int wk_top;

static int search_depth;
static int max_depth;
static int guess_count;

__attribute__((always_inline))
static inline void push_work(int cell, int dig) {
    wk_c[wk_top] = cell;
    wk_d[wk_top] = dig;
    wk_top = wk_top + 1;
}

__attribute__((always_inline))
static inline int single_digit(int cell) {
    if (cand[cell])       return 1;
    if (cand[81 + cell])  return 2;
    if (cand[162 + cell]) return 3;
    if (cand[243 + cell]) return 4;
    if (cand[324 + cell]) return 5;
    if (cand[405 + cell]) return 6;
    if (cand[486 + cell]) return 7;
    if (cand[567 + cell]) return 8;
    return 9;
}

__attribute__((always_inline))
static inline int enqueue_assign(int cell, int dig) {
    if (cand[db[dig - 1] + cell] == 0) return 0;
    if (cnt[cell] == 1) return 1;

    if (dig != 1) { if (cand[cell])       push_work(cell, 1); }
    if (dig != 2) { if (cand[81 + cell])  push_work(cell, 2); }
    if (dig != 3) { if (cand[162 + cell]) push_work(cell, 3); }
    if (dig != 4) { if (cand[243 + cell]) push_work(cell, 4); }
    if (dig != 5) { if (cand[324 + cell]) push_work(cell, 5); }
    if (dig != 6) { if (cand[405 + cell]) push_work(cell, 6); }
    if (dig != 7) { if (cand[486 + cell]) push_work(cell, 7); }
    if (dig != 8) { if (cand[567 + cell]) push_work(cell, 8); }
    if (dig != 9) { if (cand[648 + cell]) push_work(cell, 9); }

    return 1;
}

static void undo_to(int mark) {
    while (tr_top != mark) {
        tr_top = tr_top - 1;
        int cell = tr_c[tr_top];
        int dig  = tr_d[tr_top];
        int dbase = db[dig - 1];
        cand[dbase + cell] = 1;
        cnt[cell] = cnt[cell] + 1;
        int doff = d27[dig - 1];
        int u0  = cu0[cell];
        int u1  = cu1[cell];
        int u2  = cu2[cell];
        ucnt[doff + u0] = ucnt[doff + u0] + 1;
        ucnt[doff + u1] = ucnt[doff + u1] + 1;
        ucnt[doff + u2] = ucnt[doff + u2] + 1;
    }
}

static int do_elim(int cell, int dig) {
    int dbase = db[dig - 1];
    if (cand[dbase + cell] == 0) return 1;

    cand[dbase + cell] = 0;

    tr_c[tr_top] = cell;
    tr_d[tr_top] = dig;
    tr_top = tr_top + 1;

    int nc = cnt[cell] - 1;
    cnt[cell] = nc;

    int doff = d27[dig - 1];
    int u0 = cu0[cell];
    int u1 = cu1[cell];
    int u2 = cu2[cell];

    int v0 = ucnt[doff + u0] - 1; ucnt[doff + u0] = v0;
    int v1 = ucnt[doff + u1] - 1; ucnt[doff + u1] = v1;
    int v2 = ucnt[doff + u2] - 1; ucnt[doff + u2] = v2;

    if (nc == 0) return 0;

    if (nc == 1) {
        int sd = single_digit(cell);
        int sbase = db[sd - 1];
        int sdoff = d27[sd - 1];
        int uo, k, c, rem;

        rem = ucnt[sdoff + u0] - 1;
        if (rem > 0) {
            uo = unit_off[u0];
            for (k = 0; k < 9; k++) {
                c = ucells[uo + k];
                if (c != cell && cand[sbase + c]) {
                    push_work(c, sd);
                    rem = rem - 1;
                    if (rem == 0) break;
                }
            }
        }
        rem = ucnt[sdoff + u1] - 1;
        if (rem > 0) {
            uo = unit_off[u1];
            for (k = 0; k < 9; k++) {
                c = ucells[uo + k];
                if (c != cell && cand[sbase + c]) {
                    push_work(c, sd);
                    rem = rem - 1;
                    if (rem == 0) break;
                }
            }
        }
        rem = ucnt[sdoff + u2] - 1;
        if (rem > 0) {
            uo = unit_off[u2];
            for (k = 0; k < 9; k++) {
                c = ucells[uo + k];
                if (c != cell && cand[sbase + c]) {
                    push_work(c, sd);
                    rem = rem - 1;
                    if (rem == 0) break;
                }
            }
        }
    }

    if (v0 == 0) return 0;
    if (v0 == 1) {
        int uo = unit_off[u0];
        for (int k = 0; k < 9; k++) { int c = ucells[uo + k]; if (cand[dbase + c]) { if (enqueue_assign(c, dig) == 0) return 0; break; } }
    }
    if (v1 == 0) return 0;
    if (v1 == 1) {
        int uo = unit_off[u1];
        for (int k = 0; k < 9; k++) { int c = ucells[uo + k]; if (cand[dbase + c]) { if (enqueue_assign(c, dig) == 0) return 0; break; } }
    }
    if (v2 == 0) return 0;
    if (v2 == 1) {
        int uo = unit_off[u2];
        for (int k = 0; k < 9; k++) { int c = ucells[uo + k]; if (cand[dbase + c]) { if (enqueue_assign(c, dig) == 0) return 0; break; } }
    }

    return 1;
}

static unsigned char np_cell[81], np_d1[81], np_d2[81];

static int naked_pairs(void) {
    int found = 0;
    int np = 0;
    int i, j, k, d;

    for (i = 0; i < 81; i++) {
        if (cnt[i] != 2) continue;
        int a = 0, b = 0;
        for (d = 0; d < 9; d++) {
            if (cand[db[d] + i]) {
                if (a == 0) a = d + 1; else { b = d + 1; break; }
            }
        }
        np_cell[np] = i;
        np_d1[np] = a;
        np_d2[np] = b;
        np = np + 1;
    }
    if (np < 2) return 0;

    for (i = 0; i < np - 1; i++) {
        for (j = i + 1; j < np; j++) {
            if (np_d1[i] != np_d1[j]) continue;
            if (np_d2[i] != np_d2[j]) continue;
            int ci = np_cell[i], cj = np_cell[j];
            int da = np_d1[i] - 1, db_a = db[da];
            int db_idx = np_d2[i] - 1, db_b = db[db_idx];
            if (cu0[ci] == cu0[cj]) {
                int uo = unit_off[cu0[ci]];
                for (k = 0; k < 9; k++) {
                    int ck = ucells[uo + k];
                    if (ck == ci) continue;
                    if (ck == cj) continue;
                    if (cand[db_a + ck]) { push_work(ck, da + 1); found = 1; }
                    if (cand[db_b + ck]) { push_work(ck, db_idx + 1); found = 1; }
                }
            }
            if (cu1[ci] == cu1[cj]) {
                int uo = unit_off[cu1[ci]];
                for (k = 0; k < 9; k++) {
                    int ck = ucells[uo + k];
                    if (ck == ci) continue;
                    if (ck == cj) continue;
                    if (cand[db_a + ck]) { push_work(ck, da + 1); found = 1; }
                    if (cand[db_b + ck]) { push_work(ck, db_idx + 1); found = 1; }
                }
            }
            if (cu2[ci] == cu2[cj]) {
                int uo = unit_off[cu2[ci]];
                for (k = 0; k < 9; k++) {
                    int ck = ucells[uo + k];
                    if (ck == ci) continue;
                    if (ck == cj) continue;
                    if (cand[db_a + ck]) { push_work(ck, da + 1); found = 1; }
                    if (cand[db_b + ck]) { push_work(ck, db_idx + 1); found = 1; }
                }
            }
        }
    }
    return found;
}

static int propagate(void) {
    for (;;) {
        while (wk_top != 0) {
            wk_top = wk_top - 1;
            if (do_elim(wk_c[wk_top], wk_d[wk_top]) == 0) return 0;
        }
        if (naked_pairs() == 0) return 1;
    }
}

static void print_cell(int i) {
    int r = 0, c = i;
    while (c >= 9) { c = c - 9; r = r + 1; }
    printf("R%cC%c", '1' + r, '1' + c);
}

static int count_solved(void) {
    int n = 0;
    int i;
    for (i = 0; i < 81; i = i + 1) {
        if (cnt[i] <= 1) n = n + 1;
    }
    return n;
}

static int search(void) {
    int best, bc, i, mark, d;
    search_depth = search_depth + 1;
    if (search_depth > max_depth) max_depth = search_depth;

restart:
    best = -1; bc = 10;
    for (i = 0; i < 81; i++) {
        int nc = cnt[i];
        if (nc < 2) continue;
        if (nc < bc) { bc = nc; best = i; if (bc == 2) break; }
    }
    if (best < 0) {
        printf("  solved at depth %d\n", search_depth);
        search_depth = search_depth - 1;
        return 1;
    }

    mark = tr_top;

    int u0 = cu0[best];
    int u1 = cu1[best];
    int u2 = cu2[best];

    while (cnt[best] >= 2) {
        int bd = 0, bscore = -1;
        int rem = cnt[best];
        for (d = 1; d <= 9; d++) {
            if (cand[db[d - 1] + best] == 0) continue;
            int score = ucnt[d27[d - 1] + u0]
                      + ucnt[d27[d - 1] + u1]
                      + ucnt[d27[d - 1] + u2];
            if (score > bscore) { bscore = score; bd = d; }
            rem = rem - 1;
            if (rem == 0) break;
        }
        if (bd == 0) break;

        guess_count = guess_count + 1;
        printf("  d%d #%d ", search_depth, guess_count);
        print_cell(best);
        printf("=%c\n", '0' + bd);

        int pm = tr_top;
        wk_top = 0;
        if (enqueue_assign(best, bd)) {
            if (propagate()) {
                printf("    ok\n");
                if (search()) {
                    search_depth = search_depth - 1;
                    return 1;
                }
            } else {
                printf("    contradiction\n");
            }
        } else {
            printf("    eliminated\n");
        }
        undo_to(pm);
        wk_top = 0;

        printf("    elim ");
        print_cell(best);
        printf("=%c\n", '0' + bd);

        push_work(best, bd);
        if (propagate() == 0) {
            printf("    dead end, backtrack\n");
            undo_to(mark);
            search_depth = search_depth - 1;
            return 0;
        }
        if (cnt[best] <= 1) goto restart;
    }

    undo_to(mark);
    search_depth = search_depth - 1;
    return 0;
}

static void print_grid(const char *input) {
    int i;
    for (i = 0; i < 81; i = i + 1) {
        int ch = input[i];
        printf("%c", ch == '0' ? '.' : ch);
    }
    printf("\n");
}

void compute(const char *input) {
    int i, d;

    for (i = 0; i < 729; i++) cand[i] = 1;
    for (i = 0; i < 81; i++) cnt[i] = 9;
    for (i = 0; i < 243; i++) ucnt[i] = 9;

    tr_top = 0;
    wk_top = 0;
    search_depth = 0;
    max_depth = 0;
    guess_count = 0;

    printf("puzzle:\n");
    print_grid(input);
    printf("\n");

    int given = 0;
    for (i = 0; i < 81; i++) {
        int ch = input[i];
        if (ch >= '1') {
            if (ch <= '9') {
                d = ch - '0';
                if (enqueue_assign(i, d) == 0) {
                    printf("conflict!\n");
                    return;
                }
                given = given + 1;
            }
        }
    }

    printf("clues: %d\n", given);

    if (propagate() == 0) {
        printf("contradiction!\n");
        return;
    }

    int solved = count_solved();
    printf("propagated: %d/81\n\n", solved);

    if (solved < 81) {
        printf("search:\n");
        if (search() == 0) {
            printf("no solution\n");
            return;
        }
        printf("\n%d guesses, depth %d\n", guess_count, max_depth);
    } else {
        printf("solved!\n");
    }

    for (i = 0; i < 81; i++)
        printf("%c", '0' + single_digit(i));
    printf("\n");
}
