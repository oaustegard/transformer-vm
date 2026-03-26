/*
 * transformer.cpp — Standalone C++ inference for the WASM transformer.
 *
 * Loads weights from a binary file (build_model.py --save-weights=...)
 * and runs autoregressive generation with O(log n) hard attention.
 *
 * Build:  make transformer
 * Run:    ./transformer model.bin data/collatz.txt
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define HAVE_BLAS 1
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#define HAVE_BLAS 1
#elif __has_include(<cblas.h>)
extern "C" {
#include <cblas.h>
}
#define HAVE_BLAS 1
#endif

#include "hull2d_cht.h"

static int MAX_GEN = 6000000;

// ── Model ──────────────────────────────────────────────────────────────

struct Layer { const double *qkv, *out, *fi, *fo; };

struct SparseMatrix {
    std::vector<double> val;
    std::vector<int>    col;
    std::vector<int>    ptr;  // ptr[i] = start of row i in val/col
    int rows = 0, cols = 0, nnz = 0;

    void build(const double* dense, int r, int c) {
        rows = r; cols = c; nnz = 0;
        ptr.resize(r + 1);
        for (int i = 0; i < r; i++) {
            ptr[i] = nnz;
            for (int j = 0; j < c; j++) {
                if (dense[i * c + j] != 0.0) {
                    val.push_back(dense[i * c + j]);
                    col.push_back(j);
                    nnz++;
                }
            }
        }
        ptr[r] = nnz;
    }
};

#ifdef USE_SPARSE_PROJ
struct SparseLayer {
    SparseMatrix qkv, out, fi, fo;
};
#endif

struct Model {
    int V, D, L, H, F, stop;
    std::vector<std::string> name;
    std::unordered_map<std::string, int> id;
    std::vector<double> w;
    const double* emb{};
    std::vector<Layer> ly;
#ifdef USE_SPARSE_PROJ
    std::vector<SparseLayer> sly;
#endif
    const double* head{};
    SparseMatrix head_sp;
    std::vector<std::vector<int>> attn_erase;
    std::vector<std::vector<int>> ffn_erase;
    std::vector<std::vector<TieBreak>> head_tb;
};

static void load(Model& m, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int32_t h[6];
    if (fread(h, 4, 6, f) != 6) { fprintf(stderr, "bad header\n"); exit(1); }
    m.V = h[0];
    m.D = h[1];
    m.L = h[2];
    m.H = h[3];
    m.F = h[4];
    m.stop = h[5];

    m.name.resize(m.V);
    for (int i = 0; i < m.V; i++) {
        uint32_t n;
        fread(&n, 4, 1, f);
        m.name[i].resize(n);
        fread(&m.name[i][0], 1, n, f);
        m.id[m.name[i]] = i;
    }

    int D = m.D, L = m.L, F = m.F, V = m.V;
    size_t per = (size_t)(3*D*D + D*D + 2*F*D + D*F);
    size_t tot = (size_t)V*D + L*per + (size_t)V*D;
    m.w.resize(tot);
    if (fread(m.w.data(), 8, tot, f) != tot) {
        fprintf(stderr, "truncated weight file\n"); exit(1);
    }

    const double* p = m.w.data();
    m.emb = p; p += V*D;
    m.ly.resize(L);
    for (int i = 0; i < L; i++) {
        auto& l = m.ly[i];
        l.qkv = p; p += 3*D*D;
        l.out = p; p += D*D;
        l.fi  = p; p += 2*F*D;
        l.fo  = p; p += D*F;
    }
    m.head = p;

    int32_t has_erase = 0;
    if (fread(&has_erase, 4, 1, f) == 1 && has_erase) {
        m.attn_erase.resize(L);
        m.ffn_erase.resize(L);
        for (int l = 0; l < L; l++) {
            int32_t n;
            fread(&n, 4, 1, f);
            m.attn_erase[l].resize(n);
            for (int j = 0; j < n; j++) fread(&m.attn_erase[l][j], 4, 1, f);
            fread(&n, 4, 1, f);
            m.ffn_erase[l].resize(n);
            for (int j = 0; j < n; j++) fread(&m.ffn_erase[l][j], 4, 1, f);
        }
    }

    int32_t has_tb = 0;
    if (fread(&has_tb, 4, 1, f) == 1 && has_tb) {
        m.head_tb.resize(L);
        int H = m.H;
        for (int l = 0; l < L; l++) {
            m.head_tb[l].resize(H, TieBreak::AVERAGE);
            for (int h = 0; h < H; h++) {
                int32_t v;
                fread(&v, 4, 1, f);
                m.head_tb[l][h] = (v == 1) ? TieBreak::LATEST : TieBreak::AVERAGE;
            }
        }
    }
    fclose(f);

    m.head_sp.build(m.head, V, D);
    printf("Head sparsity: %d/%d nonzero (%.0f%% sparse)\n",
           m.head_sp.nnz, V * D, 100.0 * (1.0 - (double)m.head_sp.nnz / (V * D)));

#ifdef USE_SPARSE_PROJ
    m.sly.resize(L);
    long total_nnz = 0, total_dense = 0;
    for (int i = 0; i < L; i++) {
        m.sly[i].qkv.build(m.ly[i].qkv, 3*D, D);
        m.sly[i].out.build(m.ly[i].out, D,   D);
        m.sly[i].fi .build(m.ly[i].fi,  2*F, D);
        m.sly[i].fo .build(m.ly[i].fo,  D,   F);
        total_nnz   += m.sly[i].qkv.nnz + m.sly[i].out.nnz
                     + m.sly[i].fi.nnz  + m.sly[i].fo.nnz;
        total_dense += 3*D*D + D*D + 2*F*D + D*F;
    }
    printf("Projection sparsity: %ld/%ld nonzero (%.1f%% sparse)\n",
           total_nnz, total_dense,
           100.0 * (1.0 - (double)total_nnz / (double)total_dense));
#endif
}

// ── Position encoding ──────────────────────────────────────────────────

static inline void add_position_encoding(double* x, int pos) {
    x[0] += pos;
    x[1] += 1.0 / std::log(2.0) - 1.0 / std::log((double)(pos + 2));
    x[2] += (double)pos * pos;
}

// ── Linear algebra ─────────────────────────────────────────────────────

static inline void matvec(const double* __restrict__ W,
                           const double* __restrict__ x,
                           double* __restrict__ y, int rows, int cols) {
#if defined(HAVE_BLAS) && !defined(NO_BLAS)
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols,
                1.0, W, cols, x, 1, 0.0, y, 1);
#else
    for (int i = 0; i < rows; i++) {
        double s = 0;
        const double* r = W + i * cols;
        for (int j = 0; j < cols; j++) s += r[j] * x[j];
        y[i] = s;
    }
#endif
}

#ifdef USE_SPARSE_PROJ
// Sparse y = W x  using CSR form. For analytically-constructed weights
// (sparse by construction), this avoids dense matmul cost over zeros.
static inline void sparse_matvec(const SparseMatrix& W,
                                  const double* __restrict__ x,
                                  double* __restrict__ y) {
    const int rows = W.rows;
    const int* __restrict__ ptr = W.ptr.data();
    const int* __restrict__ col = W.col.data();
    const double* __restrict__ val = W.val.data();
    for (int i = 0; i < rows; i++) {
        double s = 0.0;
        const int e = ptr[i + 1];
        for (int k = ptr[i]; k < e; k++) {
            s += val[k] * x[col[k]];
        }
        y[i] = s;
    }
}
#endif

// ── Timing ─────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;
using TP = Clock::time_point;
static inline double secs(TP a, TP b) {
    return std::chrono::duration<double>(b - a).count();
}

// Per-phase timers are only compiled into the hot loop when -DPROFILE_PHASES
// is defined. Default builds have zero clock-read overhead per token.
#ifdef PROFILE_PHASES
  #define PHASE_TS(var) auto var = Clock::now()
  #define PHASE_ADD(acc, a, b) acc += secs(a, b)
#else
  #define PHASE_TS(var) ((void)0)
  #define PHASE_ADD(acc, a, b) ((void)0)
#endif

// ── Main ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.bin [--regen] [--trace[=N]] [--brute|--nohull] [--args=STR] prog1.txt [...]\n", argv[0]);
        return 1;
    }

    bool regen = false, brute = false;
    int trace_every = 0;
    long diag_at = -1;  // absolute pos in ids[] at which to emit DIAG line; -1 disables
    const char* args_str = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--regen") == 0) regen = true;
        if (strcmp(argv[i], "--brute") == 0 || strcmp(argv[i], "--nohull") == 0) brute = true;
        if (strncmp(argv[i], "--trace", 7) == 0) {
            trace_every = 1;
            if (argv[i][7] == '=') trace_every = atoi(argv[i] + 8);
        }
        if (strncmp(argv[i], "--args=", 7) == 0) args_str = argv[i] + 7;
        else if (strcmp(argv[i], "--args") == 0 && i + 1 < argc) args_str = argv[++i];
        else if (strncmp(argv[i], "--max-gen=", 10) == 0) MAX_GEN = atoi(argv[i] + 10);
        else if (strcmp(argv[i], "--max-gen") == 0 && i + 1 < argc) MAX_GEN = atoi(argv[++i]);
        else if (strncmp(argv[i], "--diag-at=", 10) == 0) diag_at = atol(argv[i] + 10);
        else if (strcmp(argv[i], "--diag-at") == 0 && i + 1 < argc) diag_at = atol(argv[++i]);
    }
    if (brute) printf("Using brute-force O(n) KV cache\n");

    Model m;
    load(m, argv[1]);
    printf("Loaded: vocab=%d D=%d layers=%d heads=%d d_ffn=%d\n",
           m.V, m.D, m.L, m.H, m.F);

    int D = m.D, F = m.F, H = m.H, L = m.L;
    int passed = 0, failed = 0, skipped = 0;
    long total_tok = 0, total_ops = 0;
    double total_time = 0;
#ifdef PROFILE_PHASES
    double t_embed = 0, t_qkv = 0, t_attn = 0, t_out = 0, t_ffn = 0, t_head = 0;
#endif
    std::vector<int> last_ids;  // save last program's tokens for batched verify

    for (int ai = 2; ai < argc; ai++) {
        if (strstr(argv[ai], "_ref")) continue;
        if (argv[ai][0] == '-' && argv[ai][1] == '-') continue;
        const char* pf = argv[ai];

        const char* bn = strrchr(pf, '/');
        bn = bn ? bn + 1 : pf;
        std::string test(bn);
        auto dot = test.rfind('.');
        if (dot != std::string::npos) test.resize(dot);

        std::ifstream fin(pf);
        if (!fin) { fprintf(stderr, "cannot open %s\n", pf); continue; }
        std::vector<int> ids;
        std::string tok;
        while (fin >> tok) {
            auto it = m.id.find(tok);
            if (it == m.id.end()) {
                fprintf(stderr, "unknown token '%s' in %s\n", tok.c_str(), pf);
                exit(1);
            }
            ids.push_back(it->second);
        }
        if (args_str) {
            int close_pos = -1;
            for (int i = (int)ids.size() - 1; i >= 0; i--)
                if (m.name[ids[i]] == "}") { close_pos = i; break; }
            if (close_pos >= 0)
                ids.resize(close_pos + 1);
            for (int i = 0; args_str[i]; i++) {
                std::string ch(1, args_str[i]);
                auto it = m.id.find(ch);
                if (it == m.id.end()) {
                    fprintf(stderr, "warning: char '%c' not in vocabulary\n", args_str[i]);
                } else {
                    ids.push_back(it->second);
                }
            }
            auto it00 = m.id.find("00");
            if (it00 != m.id.end()) ids.push_back(it00->second);
        }
        int plen = (int)ids.size();

        std::string rp(pf);
        auto dp = rp.rfind('.');
        if (dp != std::string::npos) rp.insert(dp, "_ref");
        int max_gen = MAX_GEN;
        if (!regen) {
            std::ifstream rf(rp);
            if (rf) {
                int ref_count = 0;
                std::string rt;
                while (rf >> rt) ref_count++;
                max_gen = ref_count - plen + 100;
                if (max_gen < 100) max_gen = 100;
            }
        }
        ids.reserve(plen + max_gen);

        std::vector<HardAttentionHead> hulls(brute ? 0 : L * H);
        std::vector<BruteAttentionHead> brutes(brute ? L * H : 0);
        int seq = 0;

        std::vector<double> x(D), qkv(3*D), ho(D), ao(D),
                            ff(2*F), gv(F), fo(D), logits(m.V);

        // Issue #10: when --diag-at=N is set, the forward pass at pos=N-1
        // (the step that emits ids[N]) populates this buffer with each
        // (layer, head)'s attention argmax key kx. Comparing this between
        // engines tells us whether divergence at token N originates from
        // attention picking a different historical key (attn_argmax) or
        // from head logits flipping with no upstream attn flip (head_argmax).
        std::vector<double> diag_kx(L * H, 0.0);
        bool diag_armed = (diag_at >= 0);

        printf("%s: ", test.c_str()); fflush(stdout);
        auto t0 = Clock::now();

        for (int pos = 0; pos < plen + max_gen; pos++) {
            PHASE_TS(t_pos0);
            const double* e = m.emb + ids[pos] * D;
            std::copy(e, e + D, x.data());
            add_position_encoding(x.data(), pos);
            PHASE_TS(t_pos1);
            PHASE_ADD(t_embed, t_pos0, t_pos1);

            for (int l = 0; l < L; l++) {
                const auto& ly = m.ly[l];
#ifdef USE_SPARSE_PROJ
                const auto& sly = m.sly[l];
#endif

                PHASE_TS(ta);
#ifdef USE_SPARSE_PROJ
                sparse_matvec(sly.qkv, x.data(), qkv.data());
#else
                matvec(ly.qkv, x.data(), qkv.data(), 3*D, D);
#endif
                PHASE_TS(tb);

                double *q = qkv.data(), *k = q + D, *v = k + D;
                int base = l * H;
                bool capture = diag_armed && (long)pos == diag_at - 1;
                for (int h = 0; h < H; h++) {
                    TieBreak tb = (!m.head_tb.empty()) ? m.head_tb[l][h] : TieBreak::AVERAGE;
                    double* kx_out = capture ? &diag_kx[base + h] : nullptr;
                    if (brute) {
                        brutes[base+h].insert(&k[h*2], &v[h*2], seq);
                        brutes[base+h].query(&q[h*2], tb, &ho[h*2], kx_out);
                    } else {
                        hulls[base+h].insert(&k[h*2], &v[h*2], seq);
                        hulls[base+h].query(&q[h*2], tb, &ho[h*2], kx_out);
                    }
                }
                seq++;
                PHASE_TS(tc);

#ifdef USE_SPARSE_PROJ
                sparse_matvec(sly.out, ho.data(), ao.data());
                for (int i = 0; i < D; i++) x[i] += ao[i];
#else
                matvec(ly.out, ho.data(), ao.data(), D, D);
                for (int i = 0; i < D; i++) x[i] += ao[i];
#endif
                PHASE_TS(td);

#ifdef USE_SPARSE_PROJ
                sparse_matvec(sly.fi, x.data(), ff.data());
                for (int i = 0; i < F; i++)
                    gv[i] = (ff[i] > 0 ? ff[i] : 0.0) * ff[F + i];
                sparse_matvec(sly.fo, gv.data(), fo.data());
                for (int i = 0; i < D; i++) x[i] += fo[i];
#else
                matvec(ly.fi, x.data(), ff.data(), 2*F, D);
                for (int i = 0; i < F; i++)
                    gv[i] = (ff[i] > 0 ? ff[i] : 0.0) * ff[F + i];
                matvec(ly.fo, gv.data(), fo.data(), D, F);
                for (int i = 0; i < D; i++) x[i] += fo[i];
#endif
                PHASE_TS(te);

                PHASE_ADD(t_qkv,  ta, tb);
                PHASE_ADD(t_attn, tb, tc);
                PHASE_ADD(t_out,  tc, td);
                PHASE_ADD(t_ffn,  td, te);
            }

            if (pos + 1 == (int)ids.size()) {
                PHASE_TS(th0);
                const auto& sp = m.head_sp;
                int best = 0; double bs = -1e300;
                int second = -1; double ss = -1e300;
                bool diag_now = diag_armed && (long)(pos + 1) == diag_at;
                for (int i = 0; i < sp.rows; i++) {
                    double s = 0;
                    for (int k = sp.ptr[i]; k < sp.ptr[i + 1]; k++)
                        s += sp.val[k] * x[sp.col[k]];
                    if (s > bs) { ss = bs; second = best; bs = s; best = i; }
                    else if (diag_now && s > ss)   { ss = s; second = i; }
                }
                PHASE_TS(th1);
                PHASE_ADD(t_head, th0, th1);

                if (diag_now) {
                    // One self-contained line per engine run, picked up by
                    // scripts/divergence_horizon.py for cause attribution.
                    fprintf(stderr,
                            "DIAG\t%s\t%ld\t%d\t%d\t%.17g\t%.17g",
                            test.c_str(), diag_at, best, second, bs, ss);
                    for (int lh = 0; lh < L * H; lh++)
                        fprintf(stderr, "\t%.17g", diag_kx[lh]);
                    fprintf(stderr, "\n");
                    fflush(stderr);
                    // No reason to keep running — caller only needs this
                    // single diagnostic line. Saves O(max_gen - diag_at)
                    // wasted work in the cause-attribution pass.
                    return 0;
                }

                ids.push_back(best);

                if (trace_every > 0) {
                    int gen = pos + 1 - plen;
                    if (gen % trace_every == 0 || best == m.stop) {
                        double elapsed = secs(t0, Clock::now());
                        fprintf(stderr, "[%7d %7.3fs %6.0f tok/s] %s\n",
                                gen, elapsed,
                                gen > 0 ? gen / elapsed : 0.0,
                                m.name[best].c_str());
                    }
                }

                if (best == m.stop) break;
            }
        }

#ifdef PROFILE_PHASES
        // Issue #7: dump per-head final envelope sizes (upper, lower) to stderr.
        // Prefixed lines are picked up by scripts/measure_envelope_sizes.py.
        if (!brute) {
            for (int l = 0; l < L; l++) {
                for (int h = 0; h < H; h++) {
                    int idx = l * H + h;
                    fprintf(stderr, "ENVELOPE\t%s\t%d\t%d\t%d\t%d\n",
                            test.c_str(), l, h,
                            hulls[idx].upper.size(), hulls[idx].lower.size());
                }
            }
        }
#endif

        auto t1 = Clock::now();
        double dt = secs(t0, t1);
        int nt = (int)ids.size(), no = 0;
        last_ids = ids;  // save for batched test
        std::string output_bytes;
        for (int i : ids) {
            const auto& s = m.name[i];
            if (s.find("commit") != std::string::npos || s == "branch_taken")
                no++;
            if (s.size() > 4 && s[0] == 'o' && s[1] == 'u' && s[2] == 't' && s[3] == '(') {
                unsigned byte_val;
                if (s.size() == 6 && s[5] == ')') {
                    byte_val = (unsigned char)s[4];
                } else {
                    byte_val = 0;
                    for (int j = 4; j < (int)s.size() && s[j] != ')'; j++) {
                        int d = (s[j] >= 'a') ? s[j] - 'a' + 10 : s[j] - '0';
                        byte_val = byte_val * 16 + d;
                    }
                }
                output_bytes.push_back((char)byte_val);
            }
        }
        total_tok += nt;
        total_ops += no;
        total_time += dt;

        if (regen) {
            FILE* out = fopen(rp.c_str(), "w");
            if (!out) { fprintf(stderr, "cannot write %s\n", rp.c_str()); continue; }
            bool in_prog = false;
            int prog_count = 0;
            std::vector<std::string> line_buf;
            for (int i = 0; i < nt; i++) {
                const auto& s = m.name[ids[i]];
                const char* ws = (s == " ") ? "<sp>" : s.c_str();
                if (s == "{") { fprintf(out, "{\n"); in_prog = true; prog_count = 0; continue; }
                if (s == "}") { fprintf(out, "}\n"); in_prog = false; continue; }
                if (in_prog) {
                    if (prog_count > 0) fputc(' ', out);
                    fputs(ws, out);
                    if (++prog_count == 5) { fputc('\n', out); prog_count = 0; }
                    continue;
                }
                if (!line_buf.empty()) fputc(' ', out);
                fputs(ws, out);
                line_buf.push_back(s);
                bool terminal = s.find("commit") != std::string::npos
                    || s.rfind("out(", 0) == 0 || s == "halt"
                    || s == "branch_taken" || s == "call_commit"
                    || s == "return_commit";
                if (terminal) { fputc('\n', out); line_buf.clear(); }
            }
            if (!line_buf.empty()) fputc('\n', out);
            fclose(out);
            printf("REGEN %d tok, %d ops in %.2fs (%.0f tok/s)\n",
                   nt, no, dt, dt > 0 ? nt/dt : 0.0);
            passed++;
        } else {
            std::ifstream rf(rp);
            if (rf) {
                std::vector<std::string> ref;
                std::string rt;
                while (rf >> rt) {
                    if (rt == "<sp>") rt = " ";
                    ref.push_back(rt);
                }
                int mm = -1, mn = std::min(nt, (int)ref.size());
                for (int i = 0; i < mn; i++)
                    if (m.name[ids[i]] != ref[i]) { mm = i; break; }
                if (mm >= 0) {
                    printf("  MISMATCH at %d: predicted=%s, expected=%s\n",
                           mm, m.name[ids[mm]].c_str(), ref[mm].c_str());
                    printf("FAIL (%.2fs)\n", dt);
                    failed++;
                } else if (nt < (int)ref.size()) {
                    printf("PASS  %d/%d tok, %d ops in %.2fs (%.0f tok/s) [truncated]\n",
                           nt, (int)ref.size(), no, dt, dt > 0 ? nt/dt : 0.0);
                    passed++;
                } else if (nt > (int)ref.size()) {
                    printf("PASS  %d tok (ref %d), %d ops in %.2fs (%.0f tok/s) [ref truncated]\n",
                           nt, (int)ref.size(), no, dt, dt > 0 ? nt/dt : 0.0);
                    passed++;
                } else {
                    printf("PASS  %d tok, %d ops in %.2fs (%.0f tok/s)\n",
                           nt, no, dt, dt > 0 ? nt/dt : 0.0);
                    passed++;
                }
            } else {
                printf("RAN   %d tok, %d ops in %.2fs (%.0f tok/s)\n",
                       nt, no, dt, dt > 0 ? nt/dt : 0.0);
                skipped++;
            }
            if (!output_bytes.empty()) {
                printf("  output: ");
                for (unsigned char c : output_bytes)
                    putchar((c >= 0x20 && c < 0x7f) || c == '\n' || c == '\t' ? c : '.');
                putchar('\n');
            }
        }
    }

    // ── Batched verification: dgemm projections + sequential hull ────
    // Save the last program's ids for the batched test
    if (total_tok > 0 && !brute && !last_ids.empty()) {
        int T = (int)last_ids.size();
        printf("\n── Batched verify (%d tokens) ──\n", T);

        std::vector<double> X(T * D, 0.0);
        for (int t = 0; t < T; t++) {
            const double* e = m.emb + last_ids[t] * D;
            std::copy(e, e + D, &X[t * D]);
            add_position_encoding(&X[t * D], t);
        }

        // Fresh hulls for verification
        std::vector<HardAttentionHead> vhulls(L * H);

        std::vector<double> QKV(T * 3*D), HO(T * D), AO(T * D);
        std::vector<double> FF(T * 2*F), GV(T * F), FO(T * D);
        int vseq = 0;
        double bt_proj = 0, bt_hull = 0;

        auto bstart = Clock::now();
        for (int l = 0; l < L; l++) {
            const auto& ly = m.ly[l];

            // Batch QKV: dgemm  X[T,D] @ W_qkv^T[D,3D] -> QKV[T,3D]
            auto pa = Clock::now();
#if defined(HAVE_BLAS) && !defined(NO_BLAS)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, 3*D, D, 1.0, X.data(), D, ly.qkv, D, 0.0, QKV.data(), 3*D);
#else
            for (int t = 0; t < T; t++)
                matvec(ly.qkv, &X[t*D], &QKV[t*3*D], 3*D, D);
#endif
            auto pb = Clock::now();

            // Hull: one parallel region, each thread owns a subset of heads across ALL positions
            // No barriers needed: head h's hull is independent of head h'.
            std::fill(HO.begin(), HO.end(), 0.0);
            int vseq_base = vseq;
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < H; h++) {
                TieBreak tb = (!m.head_tb.empty()) ? m.head_tb[l][h] : TieBreak::AVERAGE;
                for (int t = 0; t < T; t++) {
                    double *k = &QKV[t*3*D + D], *v = k + D, *q = &QKV[t*3*D];
                    vhulls[l*H+h].insert(&k[h*2], &v[h*2], vseq_base + t);
                    vhulls[l*H+h].query(&q[h*2], tb, &HO[t*D + h*2]);
                }
            }
            vseq += T;
            auto pc = Clock::now();

            // Batch out projection: dgemm
#if defined(HAVE_BLAS) && !defined(NO_BLAS)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, D, D, 1.0, HO.data(), D, ly.out, D, 0.0, AO.data(), D);
#else
            for (int t = 0; t < T; t++)
                matvec(ly.out, &HO[t*D], &AO[t*D], D, D);
#endif
            for (int i = 0; i < T*D; i++) X[i] += AO[i];

            // Erase attention slots
            if (!m.attn_erase.empty())
                for (int s : m.attn_erase[l])
                    for (int t = 0; t < T; t++) X[t*D + s] = 0.0;

            // Batch FFN: dgemm
#if defined(HAVE_BLAS) && !defined(NO_BLAS)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, 2*F, D, 1.0, X.data(), D, ly.fi, D, 0.0, FF.data(), 2*F);
#else
            for (int t = 0; t < T; t++)
                matvec(ly.fi, &X[t*D], &FF[t*2*F], 2*F, D);
#endif
            for (int t = 0; t < T; t++)
                for (int j = 0; j < F; j++)
                    GV[t*F+j] = (FF[t*2*F+j] > 0 ? FF[t*2*F+j] : 0.0) * FF[t*2*F+F+j];
#if defined(HAVE_BLAS) && !defined(NO_BLAS)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        T, D, F, 1.0, GV.data(), F, ly.fo, F, 0.0, FO.data(), D);
#else
            for (int t = 0; t < T; t++)
                matvec(ly.fo, &GV[t*F], &FO[t*D], D, F);
#endif
            for (int i = 0; i < T*D; i++) X[i] += FO[i];

            // Erase FFN slots
            if (!m.ffn_erase.empty())
                for (int s : m.ffn_erase[l])
                    for (int t = 0; t < T; t++) X[t*D + s] = 0.0;

            auto pd = Clock::now();
            bt_proj += secs(pa, pb) + secs(pc, pd);
            bt_hull += secs(pb, pc);
        }
        auto bend = Clock::now();
        double bt_total = secs(bstart, bend);

        // Verify: check argmax at last position
        const auto& sp = m.head_sp;
        int last = T - 2;  // second-to-last position predicts last token
        double bs = -1e300; int best = 0;
        for (int i = 0; i < sp.rows; i++) {
            double s = 0;
            for (int k = sp.ptr[i]; k < sp.ptr[i+1]; k++)
                s += sp.val[k] * X[last*D + sp.col[k]];
            if (s > bs) { bs = s; best = i; }
        }
        bool match = (best == last_ids[T-1]);

        printf("  Batched total:  %.3fs (%7.0f tok/s)\n", bt_total, T/bt_total);
        printf("  Batched proj:   %.3fs (%4.1f%%)\n", bt_proj, 100*bt_proj/bt_total);
        printf("  Batched hull:   %.3fs (%4.1f%%)\n", bt_hull, 100*bt_hull/bt_total);
        printf("  Spot check: %s (pos %d: pred=%s, actual=%s)\n",
               match ? "OK" : "MISMATCH", last,
               m.name[best].c_str(), m.name[last_ids[T-1]].c_str());
        printf("  Speedup vs sequential: %.2fx\n", total_time / bt_total);
    }

    printf("\n%d passed, %d failed, %d no-ref\n", passed, failed, skipped);
    if (total_time > 0) {
        printf("\nBenchmark: %ld tok, %ld ops, %.2fs\n"
               "  %.0f tok/s, %.0f wasm-ops/s\n",
               total_tok, total_ops, total_time,
               total_tok / total_time,
               total_ops > 0 ? total_ops / total_time : 0.0);
#ifdef PROFILE_PHASES
        double t_proj = t_qkv + t_out + t_ffn;
        double t_acc  = t_embed + t_proj + t_attn + t_head;
        double t_misc = total_time - t_acc;
        printf("\nTime breakdown:\n"
               "  embed: %.3fs (%4.1f%%)\n"
               "  qkv:   %.3fs (%4.1f%%)\n"
               "  attn:  %.3fs (%4.1f%%)\n"
               "  out:   %.3fs (%4.1f%%)\n"
               "  ffn:   %.3fs (%4.1f%%)\n"
               "  head:  %.3fs (%4.1f%%)\n"
               "  misc:  %.3fs (%4.1f%%)\n"
               "  proj:  %.3fs (%4.1f%%)   [qkv+out+ffn aggregate]\n",
               t_embed, 100*t_embed/total_time,
               t_qkv,   100*t_qkv  /total_time,
               t_attn,  100*t_attn /total_time,
               t_out,   100*t_out  /total_time,
               t_ffn,   100*t_ffn  /total_time,
               t_head,  100*t_head /total_time,
               t_misc,  100*t_misc /total_time,
               t_proj,  100*t_proj /total_time);
#endif
    }
    return failed ? 1 : 0;
}
