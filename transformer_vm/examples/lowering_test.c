/* Lowering stress test: exercises all operations that require lowering.
 * Tests: MUL, DIV_U, DIV_S, REM_U, REM_S, AND, OR, XOR, SHL, SHR_U,
 *        SHR_S, CLZ, CTZ, POPCNT, ROTL, ROTR, switch (br_table).
 *
 * Const-operand tests use direct expressions so clang emits
 * "i32.const C; OP" patterns that the lowering pass can match.
 *
 * Runtime-operand tests use optnone helpers to prevent optimization.
 *
 * Input: two decimal numbers separated by space
 * Output: results of various operations. */

__attribute__((noinline, optnone))
static int do_mul(int a, int b) { return a * b; }

__attribute__((noinline, optnone))
static unsigned do_div_u(unsigned a, unsigned b) { return a / b; }

__attribute__((noinline, optnone))
static int do_div_s(int a, int b) { return a / b; }

__attribute__((noinline, optnone))
static unsigned do_rem_u(unsigned a, unsigned b) { return a % b; }

__attribute__((noinline, optnone))
static int do_rem_s(int a, int b) { return a % b; }

__attribute__((noinline, optnone))
static int do_shl(int a, int b) { return a << b; }

__attribute__((noinline, optnone))
static unsigned do_shr_u(unsigned a, int b) { return a >> b; }

__attribute__((noinline, optnone))
static int do_shr_s(int a, int b) { return a >> b; }

__attribute__((noinline, optnone))
static int do_clz(unsigned a) { return __builtin_clz(a); }

__attribute__((noinline, optnone))
static int do_ctz(unsigned a) { return __builtin_ctz(a); }

__attribute__((noinline, optnone))
static int do_popcount(unsigned a) { return __builtin_popcount(a); }

__attribute__((noinline, optnone))
static unsigned do_rotl(unsigned a, unsigned b) {
    return __builtin_rotateleft32(a, b);
}

__attribute__((noinline, optnone))
static unsigned do_rotr(unsigned a, unsigned b) {
    return __builtin_rotateright32(a, b);
}

void compute(const char *input) {
    const char *p = input;

    int a = parse_int(p);
    while (*p >= '0' && *p <= '9') p = p + 1;
    while (*p == ' ') p = p + 1;
    int b = parse_int(p);

    printf("a=%d b=%d\n", a, b);

    /* Const-operand: MUL */
    printf("a*7: %d\n", a * 7);
    printf("a*13: %d\n", a * 13);

    /* Const-operand: DIV_U, DIV_S */
    printf("a/3u: %d\n", (int)((unsigned)a / 3));
    printf("a/10u: %d\n", (int)((unsigned)a / 10));
    printf("a/3s: %d\n", a / 3);
    printf("a/7s: %d\n", a / 7);

    /* Const-operand: REM_U, REM_S */
    printf("a%%7u: %d\n", (int)((unsigned)a % 7));
    printf("a%%10u: %d\n", (int)((unsigned)a % 10));
    printf("a%%3s: %d\n", a % 3);

    /* Const-operand: AND, OR, XOR */
    printf("a&0xFF: %d\n", a & 0xFF);
    printf("a&0x0F: %d\n", a & 0x0F);
    printf("a&1: %d\n", a & 1);
    printf("a|0x80: %d\n", a | 0x80);
    printf("a^0xFF: %d\n", a ^ 0xFF);
    printf("a^1: %d\n", a ^ 1);

    /* Const-operand: SHL, SHR_U, SHR_S */
    printf("a<<1: %d\n", a << 1);
    printf("a<<4: %d\n", a << 4);
    printf("a<<8: %d\n", a << 8);
    printf("a>>1: %d\n", (int)((unsigned)a >> 1));
    printf("a>>4: %d\n", (int)((unsigned)a >> 4));
    printf("a>>8: %d\n", (int)((unsigned)a >> 8));
    printf("a>>s8: %d\n", a >> 8);

    /* Unary: CLZ, CTZ, POPCNT */
    printf("clz(a): %d\n", do_clz((unsigned)a));
    printf("ctz(a): %d\n", do_ctz((unsigned)a));
    printf("popcnt(a): %d\n", do_popcount((unsigned)a));

    /* Const-operand: ROTL, ROTR (mask to avoid huge print_int cost) */
    printf("rotl(a,5): %d\n", (int)__builtin_rotateleft32((unsigned)a, 5));
    printf("rotr(a,5)lo: %d\n", (int)(__builtin_rotateright32((unsigned)a, 5) & 0xFF));

    /* Runtime: MUL, DIV, REM */
    printf("a*b: %d\n", do_mul(a, b));
    printf("a/bu: %d\n", (int)do_div_u((unsigned)a, (unsigned)b));
    printf("a/bs: %d\n", do_div_s(a, b));
    printf("a%%bu: %d\n", (int)do_rem_u((unsigned)a, (unsigned)b));
    printf("a%%bs: %d\n", do_rem_s(a, b));

    /* Runtime: SHL, SHR */
    printf("a<<b: %d\n", do_shl(a, b));
    printf("a>>b: %d\n", (int)do_shr_u((unsigned)a, b));
    printf("a>>sb: %d\n", do_shr_s(a, b));

    /* Runtime: CLZ, CTZ, POPCNT */
    printf("clz(b): %d\n", do_clz((unsigned)b));
    printf("ctz(b): %d\n", do_ctz((unsigned)b));
    printf("popcnt(b): %d\n", do_popcount((unsigned)b));

    /* Runtime: ROTL, ROTR (mask rotr to avoid huge print_int cost) */
    printf("rotl(a,b): %d\n", (int)do_rotl((unsigned)a, (unsigned)b));
    printf("rotr(a,b)lo: %d\n", (int)(do_rotr((unsigned)a, (unsigned)b) & 0xFF));

    /* Switch (br_table) */
    int cat = (int)((unsigned)a / 10);
    if (cat > 5) cat = 5;
    printf("category: ");
    switch (cat) {
        case 0: printf("tiny\n"); break;
        case 1: printf("small\n"); break;
        case 2: printf("medium\n"); break;
        case 3: printf("large\n"); break;
        case 4: printf("huge\n"); break;
        default: printf("massive\n"); break;
    }

    /* Combined hash */
    unsigned h = (unsigned)a;
    h = (h ^ 0xA5) * 7;
    h = ((unsigned)h >> 4) & 0xFFFF;
    printf("hash: %d\n", (int)h);

    printf("done\n");
}
