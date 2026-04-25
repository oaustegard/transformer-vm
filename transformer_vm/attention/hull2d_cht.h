#pragma once
/*
 * hull2d_cht.h — Drop-in alternative to hull2d.h
 *
 * What it does:
 *   Incremental-only 2D point structure that answers hard-attention queries:
 *     argmax_k  q · k
 *   with the same public API as hull2d.h (HardAttentionHead / BruteAttentionHead).
 *
 * Key idea (duality / 1D reduction):
 *   For q=(qx,qy) with qy != 0, maximizing q·(kx,ky) is equivalent to
 *     maximizing (kx * m + ky) for m = qx/qy  if qy > 0
 *     minimizing (kx * m + ky)               if qy < 0
 *
 *   So we maintain two dynamic 1D convex envelopes (dynamic CHT):
 *     - upper:  max   (kx*m + ky)
 *     - lower:  min   (kx*m + ky)  implemented as max of (-(kx*m + ky))
 *
 *   The upper-envelope and lower-envelope vertex sets are generally
 *   different subsets of the same point cloud (a point on the upper hull
 *   need not be on the lower hull), so both envelopes are intrinsically
 *   needed; the two cannot be folded into one without storing every
 *   line and giving up O(log h) query.
 *
 *   What we *can* fold is the per-insert allocation and metadata copy
 *   (see PointAggregate / HardAttentionHead::insert below): one
 *   per-key PointAggregate is shared between the upper and lower
 *   envelopes' Line entries, halving the metadata write traffic and
 *   eliminating the redundant duplicate-key book-keeping that
 *   `_HullCHT::add_line` used to do twice per insert. Repeat inserts
 *   of the same (kx, ky) skip both envelopes' walks entirely.
 *
 * Performance profile vs hull2d.h:
 *   - Query: O(log h) in both.
 *   - Insert:
 *       hull2d.h  : geometric walk + vector erase/insert => can cost O(h) memmove.
 *       this file : dynamic CHT => amortized O(log h) with no bulk memmove.
 *
 * Tie-breaking:
 *   Same semantics as hull2d.h: return mean of tied values (TieBreak::LATEST is
 *   accepted but behaves like AVERAGE).
 */

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════
//  PoolAlloc — thread-local free-list allocator for std::multiset nodes.
//
//  std::multiset performs one heap allocation per inserted node. For
//  CHT inserts in hot attention, that malloc latency dominates the
//  phase (jemalloc preload alone gave +11 % on a sandbox baseline,
//  which confirms malloc is the bottleneck). This allocator carves a
//  pool of fixed-size slots and replaces malloc/free with free-list
//  push/pop.
//
//  Storage is in static thread_local members. All copies of
//  PoolAlloc<T> for the same T share one pool; each rebound T has
//  its own pool. Single-threaded by construction (engine runs on
//  one thread), so no synchronization.
// ═══════════════════════════════════════════════════════════════════════

template<class T>
class PoolAlloc {
public:
    using value_type = T;

    PoolAlloc() noexcept = default;
    template<class U>
    PoolAlloc(const PoolAlloc<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n != 1) return static_cast<T*>(::operator new(n * sizeof(T)));
        if (free_head_ == nullptr) refill();
        Slot* s = free_head_;
        free_head_ = s->next;
        return reinterpret_cast<T*>(s);
    }

    void deallocate(T* p, std::size_t n) noexcept {
        if (n != 1) { ::operator delete(p); return; }
        Slot* s = reinterpret_cast<Slot*>(p);
        s->next = free_head_;
        free_head_ = s;
    }

private:
    union Slot {
        Slot* next;
        alignas(T) unsigned char raw[sizeof(T)];
    };
    static constexpr std::size_t BLOCK = 256;

    static thread_local Slot* free_head_;
    static thread_local std::vector<std::unique_ptr<Slot[]>> blocks_;

    static void refill() {
        auto block = std::make_unique<Slot[]>(BLOCK);
        for (std::size_t i = 0; i < BLOCK; i++) {
            block[i].next = free_head_;
            free_head_ = &block[i];
        }
        blocks_.push_back(std::move(block));
    }
};

template<class T> thread_local typename PoolAlloc<T>::Slot* PoolAlloc<T>::free_head_ = nullptr;
template<class T> thread_local std::vector<std::unique_ptr<typename PoolAlloc<T>::Slot[]>> PoolAlloc<T>::blocks_;

template<class T, class U>
bool operator==(const PoolAlloc<T>&, const PoolAlloc<U>&) noexcept { return true; }
template<class T, class U>
bool operator!=(const PoolAlloc<T>&, const PoolAlloc<U>&) noexcept { return false; }

enum class TieBreak { AVERAGE, LATEST };

// ═══════════════════════════════════════════════════════════════════════
//  HullMeta — aggregate value metadata (sum + count for averaging)
// ═══════════════════════════════════════════════════════════════════════

struct HullMeta {
    double vsum[2] = {0, 0};
    double vlast[2] = {0, 0};
    int    count   = 0;
    int    last_seq = -1;

    void add(const double v[2], int seq = 0) {
        vsum[0] += v[0];
        vsum[1] += v[1];
        count++;
        if (seq > last_seq) {
            last_seq = seq;
            vlast[0] = v[0];
            vlast[1] = v[1];
        }
    }

    void merge(const HullMeta& o) {
        vsum[0] += o.vsum[0];
        vsum[1] += o.vsum[1];
        count += o.count;
        if (o.last_seq > last_seq) {
            last_seq = o.last_seq;
            vlast[0] = o.vlast[0];
            vlast[1] = o.vlast[1];
        }
    }

    void resolve(TieBreak tb, double out[2]) const {
        if (count == 0) {
            out[0] = out[1] = 0;
            return;
        }
        if (tb == TieBreak::LATEST) {
            out[0] = vlast[0];
            out[1] = vlast[1];
            return;
        }
        double inv = 1.0 / count;
        out[0] = vsum[0] * inv;
        out[1] = vsum[1] * inv;
    }
};

// ═══════════════════════════════════════════════════════════════════════
//  PointAggregate — per-unique-key shared metadata block
//
//  One instance per unique (kx, ky) seen by a HardAttentionHead. Both
//  the upper envelope's Line and the lower envelope's Line for that
//  key hold a raw pointer into the head's PointAggregate pool, so any
//  metadata update (subsequent insert of the same key) is visible
//  through both envelopes immediately. The pool is owned by
//  HardAttentionHead and lives until clear().
// ═══════════════════════════════════════════════════════════════════════

struct PointAggregate {
    HullMeta meta;
};

// ═══════════════════════════════════════════════════════════════════════
//  Dynamic convex hull trick for doubles (max envelope)
//
//  This is the classic "LineContainer" approach:
//    - lines ordered by slope
//    - each line stores the x-coordinate p up to which it is optimal
//    - query uses lower_bound(x) (heterogeneous lookup) on p
//
//  Important:
//    - The set ordering is by slope (Line < Line).
//    - lower_bound(x) relies on the invariant that p is increasing with slope
//      among the *envelope* lines (maintained by insertion logic).
//
//  Lines hold a raw PointAggregate* (the value metadata) rather than
//  an inline HullMeta. The aggregate is allocated and deduplicated by
//  HardAttentionHead, so add_line() is guaranteed never to be called
//  with a duplicate (m, b) pair from this head — we do not need
//  same-line merge logic here.
// ═══════════════════════════════════════════════════════════════════════

struct _HullCHT {
    struct Line {
        double m = 0.0;   // slope
        double b = 0.0;   // intercept
        mutable long double p = 0.0L; // last x where this line is best (intersection with next)
        PointAggregate* point = nullptr;

        // Order by slope for set storage.
        bool operator<(const Line& o) const { return m < o.m; }
        // Order by breakpoint for heterogeneous queries.
        bool operator<(long double x) const { return p < x; }
    };

    using Set = std::multiset<Line, std::less<>, PoolAlloc<Line>>;
    using It  = Set::iterator;

    Set lines;

    static constexpr long double INF = std::numeric_limits<long double>::infinity();

    static inline long double eval_ld(const Line& l, long double x) {
        return (long double)l.m * x + (long double)l.b;
    }

    // Computes intersection of x and y; stores it in x->p.
    // Returns true if x->p >= y->p, meaning y is unnecessary.
    bool isect(It x, It y) {
        if (y == lines.end()) {
            x->p = INF;
            return false;
        }

        if (x->m == y->m) {
            // Keep the one with larger intercept for max envelope.
            x->p = (x->b >= y->b ? INF : -INF);
        } else {
            // x->m != y->m
            x->p = ((long double)y->b - (long double)x->b) /
                   ((long double)x->m - (long double)y->m);
        }
        return x->p >= y->p;
    }

    // Insert a line (m,b) backed by a shared PointAggregate, keeping only
    // the max envelope. The caller (HardAttentionHead) deduplicates by
    // (kx, ky) before calling, so an exact (m, b) duplicate cannot
    // occur. Same-slope-different-intercept is still possible (different
    // (kx, ky) may give the same kx → same m).
    void add_line(double m, double b, PointAggregate* point) {
        Line nl;
        nl.m = m;
        nl.b = b;
        nl.point = point;

        auto it = lines.lower_bound(nl);
        if (it != lines.end() && it->m == m) {
            if (it->b >= b) {
                // Existing dominates for all x.
                return;
            }
            // New dominates: replace.
            lines.erase(it);
        } else if (it != lines.begin()) {
            auto it2 = std::prev(it);
            if (it2->m == m) {
                if (it2->b >= b) return;
                lines.erase(it2);
            }
        }

        auto z = lines.insert(nl);
        auto y = z++;
        auto x = y;

        // Remove any lines after y that become irrelevant.
        while (isect(y, z)) z = lines.erase(z);

        // If the previous line now makes y irrelevant, erase y.
        if (x != lines.begin() && isect(--x, y)) {
            isect(x, y = lines.erase(y));
        }

        // Fix breakpoints going left.
        while ((y = x) != lines.begin() && (--x)->p >= y->p) {
            isect(x, lines.erase(y));
        }
    }

    bool empty() const { return lines.empty(); }
    int  size()  const { return (int)lines.size(); }

    void clear() { lines.clear(); }

    // Query max at x, returning iterator to best line.
    It argmax(long double x) {
        assert(!lines.empty());
        auto it = lines.lower_bound(x);
        if (it == lines.end()) it = std::prev(lines.end());
        return it;
    }

    // Const variant
    It argmax(long double x) const {
        assert(!lines.empty());
        auto it = lines.lower_bound(x);
        if (it == lines.end()) it = std::prev(lines.end());
        return it;
    }
};

// ═══════════════════════════════════════════════════════════════════════
//  HullHalf — compatible wrapper around the CHT
//
//  is_upper=true : maintains max of (kx*m + ky)
//  is_upper=false: maintains min of (kx*m + ky) via max of (-(kx*m + ky))
//                 i.e., store (m,b)=(-kx,-ky)
//
//  Lines hold a PointAggregate* shared with the sibling HullHalf, so
//  the wrapper does not own metadata storage — it only manages the
//  envelope structure and routes queries.
// ═══════════════════════════════════════════════════════════════════════

struct HullHalf {
    _HullCHT cht;
    bool is_upper;

    explicit HullHalf(bool upper = true) : is_upper(upper) {}

    int size() const { return cht.size(); }

    void clear() { cht.clear(); }

    // ─── Insert ──────────────────────────────────────────────────────
    // Adds a line corresponding to key (kx, ky) with shared metadata
    // pointer. Caller has already populated point->meta.
    void insert(double kx, double ky, PointAggregate* point) {
        if (is_upper) {
            cht.add_line(kx, ky, point);
        } else {
            // store -L so that max gives min of original
            cht.add_line(-kx, -ky, point);
        }
    }

    // ─── Query ───────────────────────────────────────────────────────
    bool query(double qx, double qy, TieBreak tb, double out[2],
               double* score_out = nullptr, double* best_kx_out = nullptr) const {
        if (cht.empty()) return false;

        if (qy == 0.0) {
            long double x = (qx >= 0 ? _HullCHT::INF : -_HullCHT::INF);
            auto it = cht.argmax(x);
            it->point->meta.resolve(tb, out);
            double kx_best = is_upper ? it->m : -it->m;
            if (score_out) {
                double ky_best = is_upper ? it->b : -it->b;
                *score_out = qx * kx_best + qy * ky_best;
            }
            if (best_kx_out) *best_kx_out = kx_best;
            return true;
        }

        long double m = (long double)qx / (long double)qy;
        auto best_it = cht.argmax(m);

        double kx_best = is_upper ? best_it->m : -best_it->m;
        double ky_best = is_upper ? best_it->b : -best_it->b;
        double best_score = qx * kx_best + qy * ky_best;

        HullMeta combined;
        combined.merge(best_it->point->meta);

        auto itL = best_it;
        while (itL != cht.lines.begin()) {
            auto prev = std::prev(itL);
            double kx_p = is_upper ? prev->m : -prev->m;
            double ky_p = is_upper ? prev->b : -prev->b;
            double s = qx * kx_p + qy * ky_p;
            if (s == best_score) {
                combined.merge(prev->point->meta);
                itL = prev;
            } else {
                break;
            }
        }

        auto itR = std::next(best_it);
        while (itR != cht.lines.end()) {
            double kx_p = is_upper ? itR->m : -itR->m;
            double ky_p = is_upper ? itR->b : -itR->b;
            double s = qx * kx_p + qy * ky_p;
            if (s == best_score) {
                combined.merge(itR->point->meta);
                ++itR;
            } else {
                break;
            }
        }

        combined.resolve(tb, out);
        if (score_out) *score_out = best_score;
        if (best_kx_out) *best_kx_out = kx_best;
        return true;
    }
};

// ═══════════════════════════════════════════════════════════════════════
//  HardAttentionHead — full 2D hard-attention KV cache (drop-in)
//
//  Maintains both envelopes (upper for qy>0, lower for qy<0) sharing a
//  single per-(kx,ky) PointAggregate pool. Repeat inserts of the same
//  key skip both envelopes' walks and only update the shared meta.
// ═══════════════════════════════════════════════════════════════════════

struct HardAttentionHead {
    HullHalf upper{true};
    HullHalf lower{false};
    HullMeta global;
    HullMeta left_meta;
    HullMeta right_meta;
    double   min_kx =  std::numeric_limits<double>::infinity();
    double   max_kx = -std::numeric_limits<double>::infinity();
    int      n = 0;

    // Shared per-(kx, ky) metadata storage. deque preserves pointer
    // stability across emplaces. Pointers handed to upper/lower's Lines
    // are valid for the lifetime of the head (or until clear()).
    std::deque<PointAggregate> point_pool;

    // Dedup: (kx, ky) → PointAggregate*. Keys live inside the head's
    // pool. Hash combines the bit patterns of the two doubles.
    struct PairHash {
        std::size_t operator()(const std::pair<double, double>& p) const noexcept {
            std::uint64_t a, b;
            std::memcpy(&a, &p.first, 8);
            std::memcpy(&b, &p.second, 8);
            // splitmix-style mix; rotate b so collisions across (a,b) vs
            // (b,a) are unlikely
            std::uint64_t h = a ^ ((b << 32) | (b >> 32));
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            return (std::size_t)h;
        }
    };
    std::unordered_map<std::pair<double, double>, PointAggregate*, PairHash> point_map;

    void clear() {
        upper.clear();
        lower.clear();
        point_map.clear();
        point_pool.clear();
        global = {};
        left_meta = {};
        right_meta = {};
        min_kx =  std::numeric_limits<double>::infinity();
        max_kx = -std::numeric_limits<double>::infinity();
        n = 0;
    }

    int size() const { return n; }

    void insert(const double key[2], const double val[2], int seq = 0) {
        global.add(val, seq);

        if (key[0] < min_kx) {
            min_kx = key[0];
            left_meta = {};
        }
        if (key[0] == min_kx) left_meta.add(val, seq);

        if (key[0] > max_kx) {
            max_kx = key[0];
            right_meta = {};
        }
        if (key[0] == max_kx) right_meta.add(val, seq);

        auto k = std::make_pair(key[0], key[1]);
        auto it = point_map.find(k);
        if (it == point_map.end()) {
            // New (kx, ky): allocate aggregate and add to both envelopes.
            point_pool.emplace_back();
            PointAggregate* p = &point_pool.back();
            p->meta.add(val, seq);
            point_map.emplace(k, p);
            upper.insert(key[0], key[1], p);
            lower.insert(key[0], key[1], p);
        } else {
            // Repeat (kx, ky): both envelopes already point at the
            // shared aggregate; just update the meta in place.
            it->second->meta.add(val, seq);
        }
        n++;
    }

    bool query(const double q[2], TieBreak tb, double out[2]) const {
        double qx = q[0], qy = q[1];
        if (qy == 0.0) {
            if (qx > 0.0)      right_meta.resolve(tb, out);
            else if (qx < 0.0) left_meta.resolve(tb, out);
            else                global.resolve(tb, out);
            return true;
        }

        double score = 0.0, best_kx = 0.0;
        bool found = false;
        if (qy > 0.0) found = upper.query(qx, qy, tb, out, &score, &best_kx);
        else               found = lower.query(qx, qy, tb, out, &score, &best_kx);

        return found;
    }
};

// ═══════════════════════════════════════════════════════════════════════
//  BruteAttentionHead — O(n) reference for verification (unchanged)
// ═══════════════════════════════════════════════════════════════════════

struct BruteAttentionHead {
    struct Entry {
        double kx, ky, vx, vy;
        int seq;
    };
    std::vector<Entry> entries;

    void clear() { entries.clear(); }
    int  size()  const { return (int)entries.size(); }

    void insert(const double key[2], const double val[2], int seq) {
        entries.push_back({key[0], key[1], val[0], val[1], seq});
    }

    bool query(const double q[2], TieBreak tb, double out[2]) const {
        if (entries.empty()) { out[0] = out[1] = 0; return false; }

        double qx = q[0], qy = q[1];
        double max_score = -std::numeric_limits<double>::infinity();
        double best_kx = 0;
        for (const auto& e : entries) {
            double s = qx * e.kx + qy * e.ky;
            if (s > max_score) { max_score = s; best_kx = e.kx; }
        }

        HullMeta meta;
        for (const auto& e : entries) {
            double s = qx * e.kx + qy * e.ky;
            if (s == max_score) {
                double v[2] = {e.vx, e.vy};
                meta.add(v, e.seq);
            }
        }
        meta.resolve(tb, out);

        return true;
    }
};
